import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from streamvggt.models.aggregator import Aggregator
from streamvggt.heads.camera_head import CameraHead
from streamvggt.heads.dpt_head import DPTHead
from streamvggt.heads.track_head import TrackHead
from transformers.file_utils import ModelOutput
from typing import Optional, Tuple, List, Any
from dataclasses import dataclass

@dataclass
class StreamVGGTOutput(ModelOutput):
    ress: Optional[List[dict]] = None
    views: Optional[torch.Tensor] = None
    memory_events: Optional[List[dict]] = None
    memory_trace: Optional[List[dict]] = None
    frame_inference_ms: Optional[List[float]] = None


def _tensor_tree_nbytes(value, seen_storages=None):
    """Count unique tensor storage bytes without synchronizing tensor values."""
    if seen_storages is None:
        seen_storages = set()
    if torch.is_tensor(value):
        storage = value.untyped_storage()
        storage_key = (str(value.device), storage.data_ptr())
        if storage_key in seen_storages:
            return 0
        seen_storages.add(storage_key)
        return storage.nbytes()
    if isinstance(value, dict):
        return sum(_tensor_tree_nbytes(item, seen_storages) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_tree_nbytes(item, seen_storages) for item in value)
    return 0


def _frame_feature(images):
    image = images[:, 0].detach().float()
    feature = F.adaptive_avg_pool2d(image, (8, 8)).flatten(1)
    return F.normalize(feature, dim=-1).mean(dim=0)


def _expand_frame_indices(frame_indices, cache_items_per_frame):
    if cache_items_per_frame == 1:
        return frame_indices
    item_offsets = torch.arange(cache_items_per_frame, device=frame_indices.device)
    return (frame_indices[:, None] * cache_items_per_frame + item_offsets[None, :]).flatten()


def _uniform_old_frame_indices(old_start, old_end, count, device):
    old_count = old_end - old_start
    if count <= 0 or old_count <= 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if count >= old_count:
        return torch.arange(old_start, old_end, device=device)
    if count == 1:
        return torch.tensor([(old_start + old_end - 1) // 2], dtype=torch.long, device=device)
    offsets = torch.div(
        torch.arange(count, device=device) * (old_count - 1),
        count - 1,
        rounding_mode="floor",
    )
    return old_start + offsets


def _split_old_recent(num_cached_frames, max_cache_frames):
    recent_count = max(1, max_cache_frames // 2)
    old_count = max_cache_frames - 1 - recent_count
    recent_start = max(1, num_cached_frames - recent_count)
    return old_count, recent_start


def _old_recent_layout(num_cached_frames, max_cache_frames, policy):
    if policy == "anchor_recent_dino_diverse_2old_1recent":
        if max_cache_frames != 4:
            raise ValueError(
                "anchor_recent_dino_diverse_2old_1recent requires "
                "cache_window_size=4"
            )
        return 2, max(1, num_cached_frames - 1)
    if policy == "anchor_recent_dino_diverse_1old_3recent":
        if max_cache_frames != 6:
            raise ValueError(
                "anchor_recent_dino_diverse_1old_3recent requires "
                "cache_window_size=6"
            )
        return 1, max(1, num_cached_frames - 4)
    return _split_old_recent(num_cached_frames, max_cache_frames)


def _different_old_frame_indices(old_start, old_end, count, device, frame_features):
    old_count = old_end - old_start
    if count <= 0 or old_count <= 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if frame_features is None or frame_features.shape[0] < old_end:
        return _uniform_old_frame_indices(old_start, old_end, count, device)

    candidates = torch.arange(old_start, old_end, device=device)
    if count >= old_count:
        return candidates

    features = frame_features.to(device=device)
    old_features = features[candidates]
    recent_features = features[old_end:]
    if recent_features.numel() == 0:
        recent_features = features[-1:]

    similarity = old_features @ recent_features.transpose(0, 1)
    scores = similarity.max(dim=1).values
    selected = candidates[torch.argsort(scores, stable=True)[:count]]
    return selected.sort().values


def _temporal_binned_dino_indices(frame_features, frame_ids, device):
    """Select an anchor, three age-binned DINO landmarks, and four recent frames."""
    if frame_features is None or frame_ids is None:
        raise ValueError("temporal_binned_dino_k8 requires DINO features and frame IDs")
    if frame_features.shape[0] != frame_ids.shape[0]:
        raise ValueError("DINO feature/frame ID count mismatch")

    current_id = frame_ids[-1]
    ages = current_id - frame_ids
    anchor = torch.nonzero(frame_ids == 0, as_tuple=False).flatten()
    if anchor.numel() == 0:
        raise ValueError("temporal_binned_dino_k8 requires frame 0 as its anchor")
    anchor = anchor[:1].to(device=device)
    recent = torch.nonzero((ages >= 0) & (ages <= 3), as_tuple=False).flatten()
    recent = recent.to(device=device)

    features = frame_features.to(device=device)
    reference = torch.cat([anchor, recent]).unique(sorted=True)
    selected_landmarks = []
    # Select from older to newer banks. The long bank uses DINO diversity.
    # Near/middle retain their oldest candidate so a landmark is guaranteed to
    # age into the next bank instead of being repeatedly replaced by a newer,
    # visually novel frame before it can provide a temporal bridge.
    for minimum_age, maximum_age in ((48, None), (16, 47), (4, 15)):
        mask = ages >= minimum_age
        if maximum_age is not None:
            mask &= ages <= maximum_age
        candidates = torch.nonzero(mask, as_tuple=False).flatten().to(device=device)
        candidates = candidates[frame_ids.index_select(0, candidates) != 0]
        if candidates.numel() == 0:
            continue
        if minimum_age == 48:
            similarities = (
                features.index_select(0, candidates)
                @ features.index_select(0, reference).transpose(0, 1)
            )
            scores = similarities.max(dim=1).values
            chosen = candidates[torch.argsort(scores, stable=True)[0:1]]
        else:
            candidate_ages = ages.index_select(0, candidates)
            chosen = candidates[torch.argsort(candidate_ages, descending=True, stable=True)[0:1]]
        selected_landmarks.append(chosen)
        reference = torch.cat([reference, chosen]).unique(sorted=True)

    selected = torch.cat([anchor, *selected_landmarks, recent]).unique(sorted=True)
    # Before middle/long banks have warmed up, use otherwise idle capacity for
    # the most recent unselected candidates. Once all banks exist this branch
    # is inactive and the steady-state layout is exactly 1+1+1+1+4 frames.
    if selected.numel() < 8:
        all_indices = torch.arange(frame_ids.shape[0], device=device)
        remaining_mask = torch.ones(frame_ids.shape[0], dtype=torch.bool, device=device)
        remaining_mask[selected] = False
        remaining = all_indices[remaining_mask]
        fill_count = min(8 - selected.numel(), remaining.numel())
        if fill_count:
            selected = torch.cat([selected, remaining[-fill_count:]])
    return selected.sort().values


def _temporal_bank_frame_ids(frame_ids):
    if frame_ids is None or frame_ids.numel() == 0:
        return {}
    current_id = int(frame_ids[-1])
    assignments = {"anchor": [], "long": [], "middle": [], "near": [], "recent": []}
    for frame_id in frame_ids.tolist():
        age = current_id - int(frame_id)
        if frame_id == 0:
            bank = "anchor"
        elif age <= 3:
            bank = "recent"
        elif age <= 15:
            bank = "near"
        elif age <= 47:
            bank = "middle"
        else:
            bank = "long"
        assignments[bank].append(int(frame_id))
    return assignments


def _candidate_similarity_log(frame_features, frame_ids, max_cache_frames, policy):
    if frame_features is None:
        return []
    if policy == "temporal_binned_dino_k8":
        return []
    if policy == "anchor_stable_adaptive_recent":
        if frame_features.shape[0] < 5:
            return []
        current_feature = frame_features[-1]
        return [
            {
                "frame_id": int(frame_ids[index]),
                "max_similarity_to_recent": float(
                    frame_features[index] @ current_feature
                ),
            }
            for index in (2, 3)
        ]
    _, recent_start = _old_recent_layout(
        frame_features.shape[0], max_cache_frames, policy
    )
    if recent_start <= 1:
        return []

    candidates = torch.arange(1, recent_start, device=frame_features.device)
    recent_features = frame_features[recent_start:]
    scores = (
        frame_features[candidates] @ recent_features.transpose(0, 1)
    ).max(dim=1).values
    candidate_ids = frame_ids.index_select(0, candidates.to(frame_ids.device))
    return [
        {
            "frame_id": int(frame_id),
            "max_similarity_to_recent": float(score),
        }
        for frame_id, score in zip(candidate_ids.tolist(), scores.tolist())
    ]


def _cache_keep_frame_indices(
    num_cached_frames,
    max_cache_frames,
    policy,
    device,
    frame_features=None,
    frame_ids=None,
    adaptive_similarity_threshold=0.99,
    adaptive_min_gap=8,
):
    if (
        policy in (
            "anchor_recent_dino_diverse_2old_1recent",
            "anchor_stable_adaptive_recent",
        )
        and max_cache_frames != 4
    ):
        raise ValueError(
            f"{policy} requires cache_window_size=4"
        )
    if (
        policy == "anchor_recent_dino_diverse_1old_3recent"
        and max_cache_frames != 6
    ):
        raise ValueError(f"{policy} requires cache_window_size=6")
    if policy == "temporal_binned_dino_k8":
        if max_cache_frames != 8:
            raise ValueError(f"{policy} requires cache_window_size=8")
        if frame_ids is None:
            raise ValueError(f"{policy} requires frame IDs")
        return _temporal_binned_dino_indices(frame_features, frame_ids, device)
    if num_cached_frames <= max_cache_frames:
        return None

    if policy == "fifo":
        return torch.arange(num_cached_frames - max_cache_frames, num_cached_frames, device=device)

    if policy == "anchor_recent":
        if max_cache_frames == 1:
            return torch.tensor([num_cached_frames - 1], device=device)
        recent = torch.arange(num_cached_frames - max_cache_frames + 1, num_cached_frames, device=device)
        return torch.cat([torch.zeros(1, dtype=torch.long, device=device), recent])

    if policy in ("anchor_recent_uniform", "anchor_recent_midpoint"):
        if max_cache_frames == 1:
            return torch.tensor([num_cached_frames - 1], device=device)
        uniform_count, recent_start = _split_old_recent(num_cached_frames, max_cache_frames)
        recent = torch.arange(recent_start, num_cached_frames, device=device)
        uniform = _uniform_old_frame_indices(1, recent_start, uniform_count, device)
        return torch.cat([torch.zeros(1, dtype=torch.long, device=device), uniform, recent])

    if policy == "anchor_recent_oldest_valid":
        if max_cache_frames == 1:
            return torch.tensor([num_cached_frames - 1], device=device)
        old_count, recent_start = _split_old_recent(num_cached_frames, max_cache_frames)
        recent = torch.arange(recent_start, num_cached_frames, device=device)
        old = torch.arange(1, min(recent_start, 1 + old_count), device=device)
        return torch.cat([torch.zeros(1, dtype=torch.long, device=device), old, recent])

    if policy == "anchor_stable_adaptive_recent":
        if frame_features is None or frame_ids is None:
            raise ValueError(f"{policy} requires DINO frame features and frame IDs")
        if num_cached_frames != 5:
            raise ValueError(
                f"{policy} expects five candidates before pruning, got {num_cached_frames}"
            )
        adaptive_index = 2
        previous_index = 3
        current_index = 4
        adaptive_similarity = float(
            frame_features[adaptive_index] @ frame_features[current_index]
        )
        adaptive_age = int(frame_ids[current_index] - frame_ids[adaptive_index])
        replace_adaptive = (
            adaptive_similarity < adaptive_similarity_threshold
            and adaptive_age >= adaptive_min_gap
        )
        selected_adaptive = previous_index if replace_adaptive else adaptive_index
        return torch.tensor(
            [0, 1, selected_adaptive, current_index],
            dtype=torch.long,
            device=device,
        )

    if policy in (
        "anchor_recent_image_diff",
        "anchor_recent_dino_diverse",
        "dino_diverse",
        "anchor_recent_dino_diverse_2old_1recent",
        "anchor_recent_dino_diverse_1old_3recent",
    ):
        if max_cache_frames == 1:
            return torch.tensor([num_cached_frames - 1], device=device)
        old_count, recent_start = _old_recent_layout(
            num_cached_frames, max_cache_frames, policy
        )
        recent = torch.arange(recent_start, num_cached_frames, device=device)
        old = _different_old_frame_indices(
            1, recent_start, old_count, device, frame_features
        )
        return torch.cat([torch.zeros(1, dtype=torch.long, device=device), old, recent])

    raise ValueError(f"Unknown cache policy: {policy}")


def _trim_kv_cache_by_frame_indices(past_key_values, frame_indices, cache_items_per_frame=1):
    if frame_indices is None:
        return past_key_values

    keep_idx = _expand_frame_indices(frame_indices, cache_items_per_frame)
    trimmed = []
    for block_kv in past_key_values:
        if block_kv is None:
            trimmed.append(None)
            continue

        k, v = block_kv
        k = k.index_select(2, keep_idx.to(k.device)).contiguous()
        v = v.index_select(2, keep_idx.to(v.device)).contiguous()
        trimmed.append((k, v))
    return trimmed


def _trim_kv_cache(
    past_key_values,
    max_cache_frames,
    policy="fifo",
    cache_items_per_frame=1,
    frame_features=None,
):
    if max_cache_frames is None:
        return past_key_values
    if max_cache_frames < 1:
        raise ValueError(f"max_cache_frames must be >= 1, got {max_cache_frames}")
    if cache_items_per_frame < 1:
        raise ValueError(f"cache_items_per_frame must be >= 1, got {cache_items_per_frame}")

    first_kv = next((block_kv for block_kv in past_key_values if block_kv is not None), None)
    if first_kv is None:
        return past_key_values

    num_cached_items = first_kv[0].shape[2]
    if num_cached_items % cache_items_per_frame != 0:
        raise ValueError(
            f"num_cached_items ({num_cached_items}) must be divisible by "
            f"cache_items_per_frame ({cache_items_per_frame})"
        )
    frame_indices = _cache_keep_frame_indices(
        num_cached_items // cache_items_per_frame,
        max_cache_frames,
        policy,
        first_kv[0].device,
        frame_features=frame_features,
    )
    return _trim_kv_cache_by_frame_indices(
        past_key_values,
        frame_indices,
        cache_items_per_frame=cache_items_per_frame,
    )


class StreamVGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size)
    


    def forward(
        self,
        views,
        query_points: torch.Tensor = None,
        history_info: Optional[dict] = None,
        past_key_values=None,
        use_cache=False,
        past_frame_idx=0
    ):
        images = torch.stack(
            [view["img"] for view in views], dim=0
        ).permute(1, 0, 2, 3, 4)    # B S C H W

        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        if history_info is None:
            history_info = {"token": None}

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)
        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

            if self.track_head is not None and query_points is not None:
                track_list, vis, conf = self.track_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
                )
                predictions["track"] = track_list[-1]  # track of the last iteration
                predictions["vis"] = vis
                predictions["conf"] = conf
            predictions["images"] = images

            B, S = images.shape[:2]
            ress = []
            for s in range(S):
                res = {
                    'pts3d_in_other_view': predictions['world_points'][:, s],  # [B, H, W, 3]
                    'conf': predictions['world_points_conf'][:, s],  # [B, H, W]

                    'depth': predictions['depth'][:, s],  # [B, H, W, 1]
                    'depth_conf': predictions['depth_conf'][:, s],  # [B, H, W]
                    'camera_pose': predictions['pose_enc'][:, s, :],  # [B, 9]

                    **({'valid_mask': views[s]["valid_mask"]}
                    if 'valid_mask' in views[s] else {}),  # [B, H, W]

                    **({'track': predictions['track'][:, s],  # [B, N, 2]
                        'vis': predictions['vis'][:, s],  # [B, N]
                        'track_conf': predictions['conf'][:, s]}
                    if 'track' in predictions else {})
                }
                ress.append(res)
            return StreamVGGTOutput(ress=ress, views=views)  # [S] [B, C, H, W]
        
    def inference(
        self,
        frames,
        query_points: torch.Tensor = None,
        past_key_values=None,
        cache_window_size: Optional[int] = None,
        cache_policy: str = "fifo",
        camera_cache_window_size: Optional[int] = None,
        camera_cache_policy: Optional[str] = None,
        return_memory_events: bool = False,
        return_memory_trace: bool = False,
        return_frame_timings: bool = False,
    ):
        if (
            cache_policy in (
                "anchor_recent_dino_diverse_2old_1recent",
                "anchor_stable_adaptive_recent",
            )
            and cache_window_size != 4
        ):
            raise ValueError(
                f"{cache_policy} requires cache_window_size=4"
            )
        if (
            cache_policy == "anchor_recent_dino_diverse_1old_3recent"
            and cache_window_size != 6
        ):
            raise ValueError(
                f"{cache_policy} requires cache_window_size=6"
            )
        if cache_policy == "temporal_binned_dino_k8" and cache_window_size != 8:
            raise ValueError(f"{cache_policy} requires cache_window_size=8")
        if camera_cache_policy is None and camera_cache_window_size is not None:
            raise ValueError(
                "camera_cache_window_size requires camera_cache_policy; "
                "omit both to keep the legacy coupled camera cache"
            )
        if camera_cache_policy == "full" and camera_cache_window_size is not None:
            raise ValueError("camera_cache_policy=full does not take a window size")
        if camera_cache_policy not in (None, "full") and camera_cache_window_size is None:
            raise ValueError(
                "an independent bounded camera cache requires "
                "camera_cache_window_size"
            )
        if camera_cache_window_size is not None and camera_cache_window_size < 1:
            raise ValueError("camera_cache_window_size must be at least 1")
        adaptive_similarity_threshold = float(
            os.environ.get("STREAMVGGT_ADAPTIVE_SIM_THRESHOLD", "0.99")
        )
        adaptive_min_gap = int(os.environ.get("STREAMVGGT_ADAPTIVE_MIN_GAP", "8"))
        if not -1.0 <= adaptive_similarity_threshold <= 1.0:
            raise ValueError("STREAMVGGT_ADAPTIVE_SIM_THRESHOLD must be in [-1, 1]")
        if adaptive_min_gap < 1:
            raise ValueError("STREAMVGGT_ADAPTIVE_MIN_GAP must be at least 1")
        if past_key_values is None:
            past_key_values = [None] * self.aggregator.depth
        past_key_values_camera = [None] * self.camera_head.trunk_depth
        cache_frame_features = None
        cache_frame_ids = None
        camera_cache_frame_ids = None
        
        all_ress = []
        processed_frames = []
        memory_events = []
        memory_trace = []
        frame_timing_events = []
        retained_output_bytes = 0
        input_tensor_bytes = _tensor_tree_nbytes(frames) if return_memory_trace else 0

        for i, frame in enumerate(frames):
            if return_frame_timings:
                timing_start = torch.cuda.Event(enable_timing=True)
                timing_end = torch.cuda.Event(enable_timing=True)
                timing_start.record()
            images = frame["img"].unsqueeze(0) 
            use_rgb_features = cache_policy == "anchor_recent_image_diff"
            use_dino_features = cache_policy in (
                "anchor_recent_dino_diverse",
                "dino_diverse",
                "anchor_recent_dino_diverse_2old_1recent",
                "anchor_recent_dino_diverse_1old_3recent",
                "temporal_binned_dino_k8",
                "anchor_stable_adaptive_recent",
            )
            if use_rgb_features:
                current_frame_feature = _frame_feature(images)
                if cache_frame_features is None:
                    cache_frame_features = current_frame_feature.unsqueeze(0)
                else:
                    cache_frame_features = torch.cat(
                        [cache_frame_features, current_frame_feature.unsqueeze(0)],
                        dim=0,
                    )
            aggregator_output = self.aggregator(
                images, 
                past_key_values=past_key_values,
                use_cache=True, 
                past_frame_idx=i,
                return_frame_features=use_dino_features,
            )
            
            if use_dino_features:
                aggregated_tokens, patch_start_idx, past_key_values, dino_features = aggregator_output
                current_frame_feature = dino_features[0, 0]
                if cache_frame_features is None:
                    cache_frame_features = current_frame_feature.unsqueeze(0)
                else:
                    cache_frame_features = torch.cat(
                        [cache_frame_features, current_frame_feature.unsqueeze(0)],
                        dim=0,
                    )
            elif isinstance(aggregator_output, tuple) and len(aggregator_output) == 3:
                aggregated_tokens, patch_start_idx, past_key_values = aggregator_output
            else:
                aggregated_tokens, patch_start_idx = aggregator_output

            current_frame_id = torch.tensor([i], dtype=torch.long, device=images.device)
            if cache_frame_ids is None:
                cache_frame_ids = current_frame_id
            else:
                cache_frame_ids = torch.cat([cache_frame_ids, current_frame_id])
            if camera_cache_frame_ids is None:
                camera_cache_frame_ids = current_frame_id
            else:
                camera_cache_frame_ids = torch.cat(
                    [camera_cache_frame_ids, current_frame_id]
                )

            keep_frame_indices = None
            if cache_window_size is not None:
                first_kv = next(
                    (block_kv for block_kv in past_key_values if block_kv is not None),
                    None,
                )
                if first_kv is not None:
                    keep_frame_indices = _cache_keep_frame_indices(
                        first_kv[0].shape[2],
                        cache_window_size,
                        cache_policy,
                        first_kv[0].device,
                        frame_features=cache_frame_features,
                        frame_ids=cache_frame_ids,
                        adaptive_similarity_threshold=adaptive_similarity_threshold,
                        adaptive_min_gap=adaptive_min_gap,
                    )
                    past_key_values = _trim_kv_cache_by_frame_indices(
                        past_key_values,
                        keep_frame_indices,
                    )
                    if keep_frame_indices is not None:
                        if return_memory_events:
                            memory_events.append(
                                {
                                    "step": i,
                                    "policy": cache_policy,
                                    "cache_window_size": cache_window_size,
                                    "candidate_frame_ids": cache_frame_ids.tolist(),
                                    "selected_frame_ids": cache_frame_ids.index_select(
                                        0, keep_frame_indices.to(cache_frame_ids.device)
                                    ).tolist(),
                                    "candidate_similarities": _candidate_similarity_log(
                                        cache_frame_features,
                                        cache_frame_ids,
                                        cache_window_size,
                                        cache_policy,
                                    ),
                                }
                            )
                        if cache_frame_features is not None:
                            cache_frame_features = cache_frame_features.index_select(
                                0, keep_frame_indices.to(cache_frame_features.device)
                            )
                        cache_frame_ids = cache_frame_ids.index_select(
                            0, keep_frame_indices.to(cache_frame_ids.device)
                        )
            
            with torch.cuda.amp.autocast(enabled=False):
                if self.camera_head is not None:
                    pose_enc, past_key_values_camera = self.camera_head(aggregated_tokens, past_key_values_camera=past_key_values_camera, use_cache=True)
                    camera_keep_frame_indices = None
                    if camera_cache_policy is None:
                        # Backward-compatible coupled mode: camera KV follows the
                        # exact same selected frames as the aggregator KV.
                        camera_keep_frame_indices = keep_frame_indices
                    elif camera_cache_policy != "full":
                        first_camera_kv = next(
                            (
                                block_kv
                                for block_kv in past_key_values_camera
                                if block_kv is not None
                            ),
                            None,
                        )
                        if first_camera_kv is not None:
                            num_camera_items = first_camera_kv[0].shape[2]
                            if num_camera_items % 4 != 0:
                                raise ValueError(
                                    "camera KV item count must be divisible by 4, "
                                    f"got {num_camera_items}"
                                )
                            camera_keep_frame_indices = _cache_keep_frame_indices(
                                num_camera_items // 4,
                                camera_cache_window_size,
                                camera_cache_policy,
                                first_camera_kv[0].device,
                                frame_ids=camera_cache_frame_ids,
                            )
                    if camera_keep_frame_indices is not None:
                        past_key_values_camera = _trim_kv_cache_by_frame_indices(
                            past_key_values_camera,
                            camera_keep_frame_indices,
                            cache_items_per_frame=4,
                        )
                        camera_cache_frame_ids = camera_cache_frame_ids.index_select(
                            0,
                            camera_keep_frame_indices.to(
                                camera_cache_frame_ids.device
                            ),
                        )
                    pose_enc = pose_enc[-1]
                    camera_pose = pose_enc[:, 0, :]

                if self.depth_head is not None:
                    depth, depth_conf = self.depth_head(
                        aggregated_tokens, images=images, patch_start_idx=patch_start_idx
                    )
                    depth = depth[:, 0] 
                    depth_conf = depth_conf[:, 0]
                
                if self.point_head is not None:
                    pts3d, pts3d_conf = self.point_head(
                        aggregated_tokens, images=images, patch_start_idx=patch_start_idx
                    )
                    pts3d = pts3d[:, 0] 
                    pts3d_conf = pts3d_conf[:, 0]

                if self.track_head is not None and query_points is not None:
                    track_list, vis, conf = self.track_head(
                        aggregated_tokens, images=images, patch_start_idx=patch_start_idx, query_points=query_points
                )
                    track = track_list[-1][:, 0]  
                    query_points = track
                    vis = vis[:, 0]
                    track_conf = conf[:, 0]

            frame_result = {
                'pts3d_in_other_view': pts3d,
                'conf': pts3d_conf,
                'depth': depth,
                'depth_conf': depth_conf,
                'camera_pose': camera_pose,
                **({'valid_mask': frame["valid_mask"]}
                    if 'valid_mask' in frame else {}),  

                **({'track': track, 
                    'vis': vis,  
                    'track_conf': track_conf}
                if query_points is not None else {})
            }
            all_ress.append(frame_result)
            if return_memory_trace:
                retained_output_bytes += _tensor_tree_nbytes(frame_result)
                memory_trace.append(
                    {
                        "frame_index": i,
                        "retained_frame_ids": cache_frame_ids.tolist(),
                        "camera_retained_frame_ids": camera_cache_frame_ids.tolist(),
                        "temporal_bank_frame_ids": (
                            _temporal_bank_frame_ids(cache_frame_ids)
                            if cache_policy == "temporal_binned_dino_k8"
                            else {}
                        ),
                        "aggregator_kv_mib": _tensor_tree_nbytes(past_key_values) / (1024 ** 2),
                        "camera_kv_mib": _tensor_tree_nbytes(past_key_values_camera) / (1024 ** 2),
                        "descriptor_mib": _tensor_tree_nbytes(cache_frame_features) / (1024 ** 2),
                        "input_tensors_mib": input_tensor_bytes / (1024 ** 2),
                        "retained_outputs_mib": retained_output_bytes / (1024 ** 2),
                        "cuda_allocated_mib": torch.cuda.memory_allocated(images.device) / (1024 ** 2),
                        "cuda_reserved_mib": torch.cuda.memory_reserved(images.device) / (1024 ** 2),
                    }
                )
            processed_frames.append(frame)
            if return_frame_timings:
                timing_end.record()
                frame_timing_events.append((timing_start, timing_end))

        frame_inference_ms = []
        if return_frame_timings:
            torch.cuda.synchronize(frames[0]["img"].device)
            frame_inference_ms = [
                float(start.elapsed_time(end)) for start, end in frame_timing_events
            ]
        
        output = StreamVGGTOutput(
            ress=all_ress,
            views=processed_frames,
            memory_events=memory_events,
            memory_trace=memory_trace,
            frame_inference_ms=frame_inference_ms,
        )
        return output
