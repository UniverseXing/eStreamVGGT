import argparse
import json
import os
import os.path as osp
import platform
import sys
import time
import traceback

import numpy as np
import open3d as o3d
import torch
from accelerate import Accelerator
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from add_ckpt_path import add_path_to_dust3r


REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "../../.."))
DEFAULT_ROOTS = {
    "7scenes": osp.join(REPO_ROOT, "data/eval/7scenes"),
    "nrgbd": osp.join(REPO_ROOT, "data/eval/neural_rgbd"),
    "eth3d": osp.join(REPO_ROOT, "data/eval/eth3d"),
    "tum": osp.join(REPO_ROOT, "data/eval/tum"),
}


def get_args_parser():
    parser = argparse.ArgumentParser("Cache-aware 3D reconstruction evaluation")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model_name", default="StreamVGGT")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--size", type=int, default=518)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("7scenes", "nrgbd", "eth3d", "tum"),
        default=("7scenes", "nrgbd", "eth3d"),
    )
    parser.add_argument(
        "--data-root",
        action="append",
        default=[],
        metavar="DATASET=PATH",
        help="override a dataset root; may be repeated",
    )
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--seq-list",
        nargs="+",
        help="evaluate only these sequence identifiers (primarily for TUM/qualitative reruns)",
    )
    parser.add_argument(
        "--dataset-seq-list",
        action="append",
        default=[],
        metavar="DATASET=SEQ1,SEQ2",
        help=(
            "filter one dataset without affecting the others; values may be "
            "comma- or whitespace-separated and the option may be repeated"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--protocol", choices=("paper", "dense"), default="paper")
    parser.add_argument("--seven-scenes-kf-every", type=int, default=50)
    parser.add_argument("--nrgbd-kf-every", type=int, default=100)
    parser.add_argument("--tum-frames", type=int, default=50)
    parser.add_argument("--tum-sampling", choices=("first", "uniform"), default="first")
    parser.add_argument("--prefix-frames", type=int, nargs="*", default=[])
    parser.add_argument("--cache-window", type=int)
    parser.add_argument("--cache-policy", default="fifo")
    parser.add_argument("--log-selections", action="store_true")
    parser.add_argument("--use_proj", action="store_true")
    parser.add_argument("--icp-threshold", type=float, default=0.1)
    parser.add_argument(
        "--save-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="save raw arrays and aligned predicted/GT point clouds",
    )
    return parser


def parse_root_overrides(values):
    roots = dict(DEFAULT_ROOTS)
    for value in values:
        if "=" not in value:
            raise ValueError(f"--data-root must be DATASET=PATH, got {value!r}")
        dataset, path = value.split("=", 1)
        dataset = dataset.lower()
        if dataset not in roots:
            raise ValueError(f"unknown dataset root override: {dataset}")
        roots[dataset] = osp.abspath(path)
    return roots


def parse_dataset_sequence_lists(values):
    selected = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"--dataset-seq-list must be DATASET=SEQ1,SEQ2, got {value!r}"
            )
        dataset, raw_sequences = value.split("=", 1)
        dataset = dataset.lower()
        if dataset not in DEFAULT_ROOTS:
            raise ValueError(f"unknown dataset sequence filter: {dataset}")
        sequences = [
            item
            for chunk in raw_sequences.split(",")
            for item in chunk.split()
            if item
        ]
        if not sequences:
            raise ValueError(f"empty sequence filter for {dataset}")
        if dataset in selected:
            raise ValueError(f"duplicate sequence filter for {dataset}")
        selected[dataset] = sequences
    return selected


def build_datasets(args, resolution, roots):
    from eval.mv_recon.data import ETH3D, NRGBD, SevenScenes, TUMDynamics

    seven_scenes_stride = 200 if args.protocol == "paper" else args.seven_scenes_kf_every
    nrgbd_stride = 500 if args.protocol == "paper" else args.nrgbd_kf_every
    builders = {
        "7scenes": lambda root: SevenScenes(
            split="test",
            ROOT=root,
            resolution=resolution,
            num_seq=1,
            full_video=True,
            kf_every=seven_scenes_stride,
            seed=args.seed,
        ),
        "nrgbd": lambda root: NRGBD(
            split="test",
            ROOT=root,
            resolution=resolution,
            num_seq=1,
            full_video=True,
            kf_every=nrgbd_stride,
            seed=args.seed,
        ),
        "eth3d": lambda root: ETH3D(
            split="test",
            ROOT=root,
            resolution=resolution,
            num_seq=1,
            num_frames=10,
            full_video=False,
            shuffle_seed=args.seed,
            seed=args.seed,
        ),
        "tum": lambda root: TUMDynamics(
            split="test",
            ROOT=root,
            resolution=resolution,
            num_frames=args.tum_frames,
            sampling=args.tum_sampling,
            seed=args.seed,
        ),
    }
    dataset_sequence_lists = parse_dataset_sequence_lists(args.dataset_seq_list)
    unknown_filters = sorted(set(dataset_sequence_lists) - set(args.datasets))
    if unknown_filters:
        raise ValueError(
            "sequence filters were provided for datasets that are not enabled: "
            f"{unknown_filters}"
        )
    datasets = {}
    for name in args.datasets:
        root = roots[name]
        if not osp.isdir(root):
            raise FileNotFoundError(f"missing {name} reconstruction root: {root}")
        datasets[name] = builders[name](root)
        dataset_specific = name in dataset_sequence_lists
        requested_sequences = dataset_sequence_lists.get(name, args.seq_list)
        if requested_sequences:
            available = list(getattr(datasets[name], "scene_list", []))
            selected_sequences = list(
                dict.fromkeys(
                    sequence
                    for sequence in requested_sequences
                    if sequence in available
                )
            )
            if dataset_specific:
                missing = [
                    sequence
                    for sequence in requested_sequences
                    if sequence not in available
                ]
                if missing:
                    raise ValueError(
                        f"requested sequences missing from {name}: {missing}; "
                        f"available examples: {available[:8]}"
                    )
            elif not selected_sequences:
                raise ValueError(
                    f"none of --seq-list {args.seq_list} exist in {name}; "
                    f"available examples: {available[:8]}"
                )
            datasets[name].scene_list = selected_sequences
            if hasattr(datasets[name], "scenes"):
                datasets[name].scenes = selected_sequences
    return datasets


def load_model(args, device):
    if args.model_name == "StreamVGGT":
        from streamvggt.models.streamvggt import StreamVGGT

        model = StreamVGGT()
    elif args.model_name == "VGGT":
        from vggt.models.vggt import VGGT

        model = VGGT()
    else:
        raise ValueError(f"unsupported model: {args.model_name}")
    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint, strict=True)
    model.eval().to(device)
    del checkpoint
    return model


def prepare_batch(sample, device, max_frames):
    batch = default_collate([sample])
    if max_frames is not None:
        batch = batch[:max_frames]
    if len(batch) < 2:
        raise ValueError("reconstruction evaluation requires at least two frames")

    gpu_keys = {"img", "camera_pose", "pts3d", "valid_mask"}
    for view in batch:
        view["img"] = (view["img"] + 1.0) / 2.0
        for key in gpu_keys:
            value = view.get(key)
            if torch.is_tensor(value):
                view[key] = value.to(device, non_blocking=True)
    return batch


def scene_id_for(dataset, data_idx):
    if hasattr(dataset, "scene_list"):
        num_seq = getattr(dataset, "num_seq", 1)
        return str(dataset.scene_list[data_idx // num_seq])
    return str(data_idx)


def safe_scene_name(scene_id):
    return scene_id.replace("/", "_").replace("\\", "_")


def center_crop_224(array):
    height, width = array.shape[:2]
    cx, cy = width // 2, height // 2
    return array[cy - 112 : cy + 112, cx - 112 : cx + 112]


def umeyama_alignment(src, dst):
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"expected matching Nx3 points, got {src.shape} and {dst.shape}")
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_centered = src - mu_src
    dst_centered = dst - mu_dst
    covariance = dst_centered.T @ src_centered / len(src)
    u, singular_values, vh = np.linalg.svd(covariance)
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vh) < 0:
        sign[-1, -1] = -1
    rotation = u @ sign @ vh
    variance = np.square(src_centered).sum() / len(src)
    scale = (singular_values * np.diag(sign)).sum() / variance
    translation = mu_dst - scale * rotation @ mu_src
    return scale, rotation, translation


def extract_pointclouds(eval_batch, preds, use_proj, projected_points=None):
    from eval.mv_recon.criterion import L21, Regr3D_t_ScaleShiftInv

    criterion = Regr3D_t_ScaleShiftInv(L21, norm_mode=False, gt_scale=True)
    gt_pts, pred_pts, _, _, _, _ = criterion.get_all_pts3d_t(eval_batch, preds)

    images_all = []
    points_all = []
    gt_points_all = []
    masks_all = []
    for index, view in enumerate(eval_batch):
        image = view["img"].permute(0, 2, 3, 1).detach().cpu().numpy()[0]
        valid = view["valid_mask"].detach().cpu().numpy()[0]
        points = projected_points[index] if use_proj else pred_pts[index].detach().cpu().numpy()[0]
        gt_points = gt_pts[index].detach().cpu().numpy()[0]
        images_all.append(center_crop_224(image)[None])
        points_all.append(center_crop_224(points)[None])
        gt_points_all.append(center_crop_224(gt_points)[None])
        masks_all.append(center_crop_224(valid)[None])
    return (
        np.concatenate(images_all),
        np.concatenate(points_all),
        np.concatenate(gt_points_all),
        np.concatenate(masks_all),
    )


def evaluate_pointclouds(images, points, gt_points, valid_masks, icp_threshold, use_proj):
    from eval.mv_recon.utils import accuracy, completion

    pred_valid = valid_masks.astype(bool) & np.isfinite(points).all(axis=-1)
    gt_valid = valid_masks.astype(bool) & np.isfinite(gt_points).all(axis=-1)
    if use_proj:
        common = pred_valid & gt_valid
        source = points[common].reshape(-1, 3)
        target = gt_points[common].reshape(-1, 3)
        scale, rotation, translation = umeyama_alignment(source, target)
        points = scale * np.einsum("ij,...j->...i", rotation, points) + translation
        pred_valid = valid_masks.astype(bool) & np.isfinite(points).all(axis=-1)

    pred_xyz = points[pred_valid].reshape(-1, 3)
    gt_xyz = gt_points[gt_valid].reshape(-1, 3)
    pred_rgb = images[pred_valid].reshape(-1, 3)
    gt_rgb = images[gt_valid].reshape(-1, 3)
    if len(pred_xyz) == 0 or len(gt_xyz) == 0:
        raise ValueError("no finite valid points for reconstruction evaluation")

    predicted = o3d.geometry.PointCloud()
    predicted.points = o3d.utility.Vector3dVector(pred_xyz)
    predicted.colors = o3d.utility.Vector3dVector(pred_rgb)
    ground_truth = o3d.geometry.PointCloud()
    ground_truth.points = o3d.utility.Vector3dVector(gt_xyz)
    ground_truth.colors = o3d.utility.Vector3dVector(gt_rgb)

    registration = o3d.pipelines.registration.registration_icp(
        predicted,
        ground_truth,
        icp_threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    predicted.transform(registration.transformation)
    predicted.estimate_normals()
    ground_truth.estimate_normals()

    gt_normals = np.asarray(ground_truth.normals)
    pred_normals = np.asarray(predicted.normals)
    acc, acc_med, nc1, nc1_med = accuracy(
        ground_truth.points, predicted.points, gt_normals, pred_normals
    )
    comp, comp_med, nc2, nc2_med = completion(
        ground_truth.points, predicted.points, gt_normals, pred_normals
    )
    metrics = {
        "acc": float(acc),
        "acc_med": float(acc_med),
        "comp": float(comp),
        "comp_med": float(comp_med),
        "nc1": float(nc1),
        "nc1_med": float(nc1_med),
        "nc2": float(nc2),
        "nc2_med": float(nc2_med),
        "nc": float((nc1 + nc2) / 2.0),
        "nc_med": float((nc1_med + nc2_med) / 2.0),
        "overall": float((acc + comp) / 2.0),
        "icp_fitness": float(registration.fitness),
        "icp_rmse": float(registration.inlier_rmse),
    }
    return metrics, predicted, ground_truth


PREFIX_METRIC_KEYS = (
    "acc",
    "acc_med",
    "comp",
    "comp_med",
    "nc",
    "nc_med",
    "overall",
)


def evaluate_prefixes(
    eval_batch,
    preds,
    projected_points,
    requested_prefixes,
    frame_timings,
    icp_threshold,
    use_proj,
    full_metrics,
):
    prefix_results = []
    for prefix in sorted(set(requested_prefixes)):
        if prefix < 2 or prefix > len(preds):
            continue
        if prefix == len(preds):
            metrics = full_metrics
        else:
            prefix_projected = (
                projected_points[:prefix] if projected_points is not None else None
            )
            images, points, gt_points, valid_masks = extract_pointclouds(
                eval_batch[:prefix],
                preds[:prefix],
                use_proj,
                prefix_projected,
            )
            metrics, _, _ = evaluate_pointclouds(
                images,
                points,
                gt_points,
                valid_masks,
                icp_threshold,
                use_proj,
            )
        prefix_results.append(
            {
                "prefix_frames": prefix,
                "final_frame_ms": (
                    float(frame_timings[prefix - 1]) if frame_timings else None
                ),
                **{key: metrics[key] for key in PREFIX_METRIC_KEYS},
            }
        )
    return prefix_results


def mean_or_none(rows, key):
    values = [
        row[key]
        for row in rows
        if row.get("status") == "ok" and row.get(key) is not None
    ]
    return float(np.mean(values)) if values else None


def summarize_prefixes(dataset_name, rows, cache_window, cache_policy, protocol):
    grouped = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        for prefix_result in row.get("prefix_metrics", []):
            grouped.setdefault(prefix_result["prefix_frames"], []).append(prefix_result)
    summaries = []
    for prefix, prefix_rows in sorted(grouped.items()):
        summary = {
            "dataset": dataset_name,
            "protocol": protocol,
            "prefix_frames": prefix,
            "cache_window_size": cache_window,
            "cache_policy": "full_cache" if cache_window is None else cache_policy,
            "num_sequences": len(prefix_rows),
        }
        for key in PREFIX_METRIC_KEYS:
            summary[f"mean_{key}"] = float(
                np.mean([prefix_row[key] for prefix_row in prefix_rows])
            )
        timings = [
            prefix_row["final_frame_ms"]
            for prefix_row in prefix_rows
            if prefix_row["final_frame_ms"] is not None
        ]
        summary["mean_final_frame_ms"] = float(np.mean(timings)) if timings else None
        summaries.append(summary)
    return summaries


def summarize_dataset(dataset_name, rows, cache_window, cache_policy, protocol, stride):
    successful = [row for row in rows if row["status"] == "ok"]
    failed = [row for row in rows if row["status"] != "ok"]
    total_frames = sum(row["num_frames"] for row in successful)
    total_time = sum(row["inference_sec"] for row in successful)
    return {
        "dataset": dataset_name,
        "protocol": protocol,
        "sampling_stride": stride,
        "prefix_frames": None,
        "cache_window_size": cache_window,
        "cache_policy": "full_cache" if cache_window is None else cache_policy,
        "num_sequences": len(rows),
        "num_successful": len(successful),
        "num_failed": len(failed),
        "total_frames": total_frames,
        "mean_acc": mean_or_none(rows, "acc"),
        "mean_acc_med": mean_or_none(rows, "acc_med"),
        "mean_comp": mean_or_none(rows, "comp"),
        "mean_comp_med": mean_or_none(rows, "comp_med"),
        "mean_nc": mean_or_none(rows, "nc"),
        "mean_nc_med": mean_or_none(rows, "nc_med"),
        "mean_overall": mean_or_none(rows, "overall"),
        "mean_ate": mean_or_none(rows, "ate"),
        "mean_rpe_trans": mean_or_none(rows, "rpe_trans"),
        "mean_rpe_rot_deg": mean_or_none(rows, "rpe_rot_deg"),
        "total_inference_sec": total_time,
        "fps_inference": total_frames / total_time if total_time else None,
        "mean_final_frame_ms": mean_or_none(rows, "final_frame_ms"),
        "max_peak_allocated_mb": max(
            (row["peak_allocated_mb"] for row in successful), default=None
        ),
        "max_peak_reserved_mb": max(
            (row["peak_reserved_mb"] for row in successful), default=None
        ),
    }


def write_legacy_log(path, rows, summary):
    with open(path, "w") as handle:
        for row in rows:
            if row["status"] == "ok":
                handle.write(
                    f"Idx: {row['sequence']}, Acc: {row['acc']}, Comp: {row['comp']}, "
                    f"NC1: {row['nc1']}, NC2: {row['nc2']} - "
                    f"Acc_med: {row['acc_med']}, Compc_med: {row['comp_med']}, "
                    f"NC1c_med: {row['nc1_med']}, NC2c_med: {row['nc2_med']}\n"
                )
            else:
                handle.write(f"FAILED {row['sequence']}: {row['error']}\n")
        handle.write("mean: " + json.dumps(summary, sort_keys=True) + "\n")


def main(args):
    if args.max_scenes is not None and args.max_scenes < 1:
        raise ValueError("--max-scenes must be at least 1")
    if args.max_frames is not None and args.max_frames < 2:
        raise ValueError("--max-frames must be at least 2")
    if args.cache_window is not None and args.cache_window < 1:
        raise ValueError("--cache-window must be at least 1")
    if args.seven_scenes_kf_every < 1 or args.nrgbd_kf_every < 1:
        raise ValueError("dense reconstruction strides must be at least 1")
    if args.tum_frames < 2:
        raise ValueError("--tum-frames must be at least 2")
    if any(prefix < 2 for prefix in args.prefix_frames):
        raise ValueError("--prefix-frames values must be at least 2")

    add_path_to_dust3r(args.weights)
    if args.size == 518:
        resolution = (518, 392)
    elif args.size == 512:
        resolution = (512, 384)
    elif args.size == 224:
        resolution = 224
    else:
        raise ValueError(f"unsupported image size: {args.size}")

    roots = parse_root_overrides(args.data_root)
    datasets = build_datasets(args, resolution, roots)
    accelerator = Accelerator()
    if accelerator.num_processes != 1:
        raise RuntimeError("structured reconstruction output currently requires one process")
    device = accelerator.device
    if device.type != "cuda":
        raise RuntimeError("3D reconstruction evaluation requires CUDA")
    model = load_model(args, device)

    if args.use_proj:
        from streamvggt.utils.geometry import unproject_depth_map_to_point_map
    from eval.pose_evaluation.trajectory_metrics import evaluate_trajectory
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

    os.makedirs(args.output_dir, exist_ok=True)
    sampling_strides = {
        "7scenes": 200 if args.protocol == "paper" else args.seven_scenes_kf_every,
        "nrgbd": 500 if args.protocol == "paper" else args.nrgbd_kf_every,
        "eth3d": "random_10",
        "tum": f"{args.tum_sampling}_{args.tum_frames}",
    }
    payload = {
        "model_name": args.model_name,
        "gpu_name": torch.cuda.get_device_name(device),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda or "",
        "python_version": platform.python_version(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": platform.node(),
        "input_size": args.size,
        "use_proj": args.use_proj,
        "max_scenes": args.max_scenes,
        "max_frames": args.max_frames,
        "icp_threshold": args.icp_threshold,
        "protocol": args.protocol,
        "sampling_strides": sampling_strides,
        "prefix_frames": sorted(set(args.prefix_frames)),
        "cache_window_size": args.cache_window,
        "cache_policy": "full_cache" if args.cache_window is None else args.cache_policy,
        "seed": args.seed,
        "datasets": {},
    }

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    total_successful = 0
    failed_datasets = []
    for dataset_name, dataset in datasets.items():
        dataset_dir = osp.join(args.output_dir, dataset_name)
        selection_dir = osp.join(dataset_dir, "memory_selections")
        trajectory_dir = osp.join(dataset_dir, "trajectories")
        os.makedirs(dataset_dir, exist_ok=True)
        os.makedirs(trajectory_dir, exist_ok=True)
        if args.log_selections:
            os.makedirs(selection_dir, exist_ok=True)

        indices = list(range(len(dataset)))
        if args.max_scenes is not None:
            indices = indices[: args.max_scenes]
        rows = []
        for data_idx in tqdm(indices, desc=dataset_name):
            scene_id = scene_id_for(dataset, data_idx)
            scene_key = safe_scene_name(scene_id)
            print(f"Evaluation for {dataset_name}/{scene_id}")
            batch = output = preds = eval_batch = projected_points = None
            pose_encoding = depth_map = extrinsic = intrinsic = None
            pred_w2c = pred_c2w = gt_c2w = aligned_c2w = None
            images = points = gt_points = valid_masks = None
            predicted_pcd = gt_pcd = None
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                batch = prepare_batch(dataset[data_idx], device, args.max_frames)
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
                    if args.model_name == "StreamVGGT":
                        output = model.inference(
                            batch,
                            cache_window_size=args.cache_window,
                            cache_policy=args.cache_policy,
                            return_memory_events=args.log_selections,
                            return_frame_timings=True,
                        )
                    else:
                        output = model.inference(batch)
                torch.cuda.synchronize(device)
                inference_sec = time.perf_counter() - start
                peak_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
                preds, eval_batch = output.ress, output.views

                pose_encoding = torch.stack(
                    [prediction["camera_pose"] for prediction in preds], dim=1
                )
                if args.use_proj:
                    depth_map = torch.stack(
                        [prediction["depth"] for prediction in preds], dim=1
                    )
                    extrinsic, intrinsic = pose_encoding_to_extri_intri(
                        pose_encoding, eval_batch[0]["img"].shape[-2:]
                    )
                    projected_points = unproject_depth_map_to_point_map(
                        depth_map.squeeze(0), extrinsic.squeeze(0), intrinsic.squeeze(0)
                    )
                else:
                    extrinsic, _ = pose_encoding_to_extri_intri(
                        pose_encoding,
                        eval_batch[0]["img"].shape[-2:],
                        build_intrinsics=False,
                    )

                pred_w2c = extrinsic[0].detach().cpu().double().numpy()
                pred_w2c_h = np.tile(np.eye(4, dtype=np.float64), (len(pred_w2c), 1, 1))
                pred_w2c_h[:, :3] = pred_w2c
                pred_c2w = np.linalg.inv(pred_w2c_h)
                gt_c2w = np.stack(
                    [view["camera_pose"][0].detach().cpu().double().numpy() for view in eval_batch]
                )
                trajectory_path = osp.join(trajectory_dir, f"{scene_key}.npz")
                np.savez_compressed(
                    trajectory_path,
                    gt_c2w=gt_c2w,
                    pred_c2w=pred_c2w,
                )
                pose_metrics = {
                    "pose_status": "failed",
                    "pose_error": None,
                    "ate": None,
                    "rpe_trans": None,
                    "rpe_rot_deg": None,
                }
                try:
                    evaluated_pose = evaluate_trajectory(gt_c2w, pred_c2w)
                    aligned_c2w = evaluated_pose.pop("pred_c2w_aligned")
                    pose_metrics = {
                        "pose_status": "ok",
                        "pose_error": None,
                        **evaluated_pose,
                    }
                    np.savez_compressed(
                        trajectory_path,
                        gt_c2w=gt_c2w,
                        pred_c2w=pred_c2w,
                        pred_c2w_aligned=aligned_c2w,
                    )
                except Exception as pose_error:
                    pose_metrics["pose_error"] = (
                        f"{type(pose_error).__name__}: {pose_error}"
                    )
                    print(
                        f"Pose metric unavailable for {dataset_name}/{scene_id}: "
                        f"{pose_metrics['pose_error']}",
                        file=sys.stderr,
                    )

                images, points, gt_points, valid_masks = extract_pointclouds(
                    eval_batch, preds, args.use_proj, projected_points
                )
                metrics, predicted_pcd, gt_pcd = evaluate_pointclouds(
                    images,
                    points,
                    gt_points,
                    valid_masks,
                    args.icp_threshold,
                    args.use_proj,
                )
                frame_timings = getattr(output, "frame_inference_ms", None) or []
                prefix_metrics = evaluate_prefixes(
                    eval_batch,
                    preds,
                    projected_points,
                    args.prefix_frames,
                    frame_timings,
                    args.icp_threshold,
                    args.use_proj,
                    metrics,
                )
                result = {
                    "dataset": dataset_name,
                    "sequence": scene_id,
                    "status": "ok",
                    "num_frames": len(preds),
                    "inference_sec": inference_sec,
                    "fps_inference": len(preds) / inference_sec,
                    "mean_frame_ms": float(np.mean(frame_timings)) if frame_timings else None,
                    "final_frame_ms": float(frame_timings[-1]) if frame_timings else None,
                    "peak_allocated_mb": peak_allocated,
                    "peak_reserved_mb": peak_reserved,
                    "prefix_metrics": prefix_metrics,
                    **pose_metrics,
                    **metrics,
                }
                rows.append(result)
                print(json.dumps(result, sort_keys=True))

                if args.save_artifacts:
                    np.save(
                        osp.join(dataset_dir, f"{scene_key}.npy"),
                        {
                            "images_all": images,
                            "pts_all": points,
                            "pts_gt_all": gt_points,
                            "masks_all": valid_masks,
                        },
                    )
                    o3d.io.write_point_cloud(
                        osp.join(dataset_dir, f"{scene_key}-mask_align.ply"), predicted_pcd
                    )
                    o3d.io.write_point_cloud(
                        osp.join(dataset_dir, f"{scene_key}-gt.ply"), gt_pcd
                    )
                if args.log_selections and getattr(output, "memory_events", None):
                    with open(osp.join(selection_dir, f"{scene_key}.json"), "w") as handle:
                        json.dump(output.memory_events, handle, indent=2)
            except Exception as error:
                failure = {
                    "dataset": dataset_name,
                    "sequence": scene_id,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
                rows.append(failure)
                print(json.dumps(failure), file=sys.stderr)
                traceback.print_exc()
            finally:
                batch = output = preds = eval_batch = projected_points = None
                pose_encoding = depth_map = extrinsic = intrinsic = None
                pred_w2c = pred_c2w = gt_c2w = aligned_c2w = None
                images = points = gt_points = valid_masks = None
                predicted_pcd = gt_pcd = None
                torch.cuda.empty_cache()

        summary = summarize_dataset(
            dataset_name,
            rows,
            args.cache_window,
            args.cache_policy,
            args.protocol,
            sampling_strides[dataset_name],
        )
        prefix_summaries = summarize_prefixes(
            dataset_name,
            rows,
            args.cache_window,
            args.cache_policy,
            args.protocol,
        )
        total_successful += summary["num_successful"]
        if summary["num_successful"] == 0:
            failed_datasets.append(dataset_name)
        dataset_payload = {
            "summary": summary,
            "prefix_summaries": prefix_summaries,
            "sequences": rows,
        }
        payload["datasets"][dataset_name] = dataset_payload
        with open(osp.join(dataset_dir, "metrics.json"), "w") as handle:
            json.dump(dataset_payload, handle, indent=2)
        write_legacy_log(osp.join(dataset_dir, "logs_all.txt"), rows, summary)
        with open(osp.join(args.output_dir, "reconstruction_metrics.json"), "w") as handle:
            json.dump(payload, handle, indent=2)
        print(json.dumps(summary, indent=2))

    if total_successful == 0:
        raise RuntimeError("all reconstruction sequences failed")
    if failed_datasets:
        raise RuntimeError(
            "all sequences failed for reconstruction dataset(s): "
            + ", ".join(failed_datasets)
        )


if __name__ == "__main__":
    main(get_args_parser().parse_args())
