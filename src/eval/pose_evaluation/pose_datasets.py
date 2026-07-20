import glob
import os

import numpy as np
from scipy.spatial.transform import Rotation


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

DEFAULT_ROOTS = {
    "sintel": os.path.join(REPO_ROOT, "data/eval/sintel/training/final"),
    "scannet": os.path.join(REPO_ROOT, "data/eval/scannetv2"),
    "tum": os.path.join(REPO_ROOT, "data/eval/tum"),
}

DEFAULT_ANNO_ROOTS = {
    "sintel": os.path.join(REPO_ROOT, "data/eval/sintel/training/camdata_left"),
}

# Keep the same 14-sequence Sintel subset used by the repository's existing
# video-depth evaluation instead of silently evaluating all training clips.
SINTEL_EVAL_SEQUENCES = (
    "alley_2",
    "ambush_4",
    "ambush_5",
    "ambush_6",
    "cave_2",
    "cave_4",
    "market_2",
    "market_5",
    "market_6",
    "shaman_3",
    "sleeping_1",
    "sleeping_2",
    "temple_2",
    "temple_3",
)


def discover_sequences(dataset, data_root, anno_root=None):
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"missing {dataset} data root: {data_root}")
    candidates = SINTEL_EVAL_SEQUENCES if dataset == "sintel" else sorted(os.listdir(data_root))
    sequences = []
    for name in candidates:
        path = os.path.join(data_root, name)
        if not os.path.isdir(path):
            continue
        if dataset == "sintel":
            if anno_root and os.path.isdir(os.path.join(anno_root, name)):
                sequences.append(name)
        elif dataset == "scannet":
            if os.path.isdir(os.path.join(path, "color_90")) and os.path.isfile(
                os.path.join(path, "pose_90.txt")
            ):
                sequences.append(name)
        elif dataset == "tum":
            if os.path.isdir(os.path.join(path, "rgb_90")) and os.path.isfile(
                os.path.join(path, "groundtruth_90.txt")
            ):
                sequences.append(name)
    return sequences


def _homogeneous(matrix_3x4):
    result = np.eye(4, dtype=np.float64)
    result[:3] = matrix_3x4
    return result


def read_sintel_cam(path):
    with open(path, "rb") as handle:
        check = np.fromfile(handle, dtype=np.float32, count=1)[0]
        if check != 202021.25:
            raise ValueError(f"invalid Sintel camera tag in {path}: {check}")
        np.fromfile(handle, dtype=np.float64, count=9)
        world_to_camera = np.fromfile(handle, dtype=np.float64, count=12).reshape(3, 4)
    return np.linalg.inv(_homogeneous(world_to_camera))


def read_matrix_trajectory(path):
    values = np.loadtxt(path, comments="#", dtype=np.float64)
    values = np.atleast_2d(values)
    if values.shape[1] == 16:
        poses = values.reshape(-1, 4, 4)
    elif values.shape[1] == 12:
        poses = np.stack([_homogeneous(row.reshape(3, 4)) for row in values])
    elif values.shape[1] == 4 and values.shape[0] % 4 == 0:
        poses = values.reshape(-1, 4, 4)
    else:
        raise ValueError(f"unsupported matrix trajectory shape {values.shape} in {path}")
    return poses


def read_tum_trajectory(path):
    values = np.loadtxt(path, comments="#", dtype=np.float64)
    values = np.atleast_2d(values)
    if values.shape[1] != 8:
        raise ValueError(f"expected TUM trajectory with 8 columns in {path}, got {values.shape}")
    poses = []
    for row in values:
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = Rotation.from_quat(row[4:8]).as_matrix()
        pose[:3, 3] = row[1:4]
        poses.append(pose)
    return np.stack(poses)


def _pair_and_sample(image_paths, poses, stride, max_frames, sequence):
    if len(image_paths) != len(poses):
        raise ValueError(
            f"{sequence}: image/pose count mismatch ({len(image_paths)} vs {len(poses)}); "
            "use the aligned *_90 preprocessing expected by MonST3R"
        )
    valid = np.isfinite(poses).all(axis=(1, 2))
    pairs = [(path, pose) for path, pose, keep in zip(image_paths, poses, valid) if keep]
    pairs = pairs[::stride]
    if max_frames is not None:
        pairs = pairs[:max_frames]
    if len(pairs) < 2:
        raise ValueError(f"{sequence}: fewer than two valid aligned frames")
    paths, sampled_poses = zip(*pairs)
    return list(paths), np.stack(sampled_poses)


def load_pose_sequence(
    dataset,
    data_root,
    sequence,
    anno_root=None,
    stride=1,
    max_frames=None,
):
    if dataset == "sintel":
        image_paths = sorted(glob.glob(os.path.join(data_root, sequence, "*.png")))
        camera_paths = [
            os.path.join(anno_root, sequence, os.path.splitext(os.path.basename(path))[0] + ".cam")
            for path in image_paths
        ]
        missing = [path for path in camera_paths if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError(f"missing Sintel camera file: {missing[0]}")
        poses = np.stack([read_sintel_cam(path) for path in camera_paths])
    elif dataset == "scannet":
        image_paths = sorted(glob.glob(os.path.join(data_root, sequence, "color_90", "*.jpg")))
        poses = read_matrix_trajectory(os.path.join(data_root, sequence, "pose_90.txt"))
    elif dataset == "tum":
        image_paths = sorted(glob.glob(os.path.join(data_root, sequence, "rgb_90", "*")))
        image_paths = [
            path for path in image_paths if os.path.splitext(path)[1].lower() in (".png", ".jpg", ".jpeg")
        ]
        poses = read_tum_trajectory(os.path.join(data_root, sequence, "groundtruth_90.txt"))
    else:
        raise ValueError(f"unsupported pose dataset: {dataset}")
    return _pair_and_sample(image_paths, poses, stride, max_frames, sequence)
