import copy

import numpy as np
from evo.core import metrics
from evo.core.metrics import PoseRelation, StatisticsType
from evo.core.trajectory import PosePath3D
from evo.core.units import Unit


def _project_rotations_to_so3(poses):
    """Return SE(3) poses whose rotations are the nearest proper rotations."""
    projected = np.array(poses, dtype=np.float64, copy=True)
    rotations = projected[:, :3, :3]
    u, _, vh = np.linalg.svd(rotations)
    nearest = u @ vh
    reflected = np.linalg.det(nearest) < 0
    if np.any(reflected):
        u[reflected, :, -1] *= -1
        nearest = u @ vh
    correction = np.linalg.norm(rotations - nearest, axis=(1, 2))
    projected[:, :3, :3] = nearest
    projected[:, 3] = np.array([0.0, 0.0, 0.0, 1.0])
    return projected, float(correction.max(initial=0.0))


def evaluate_trajectory(gt_c2w, pred_c2w):
    """Evaluate camera-to-world trajectories after Sim(3) alignment."""
    gt_c2w = np.asarray(gt_c2w, dtype=np.float64)
    pred_c2w = np.asarray(pred_c2w, dtype=np.float64)
    if gt_c2w.shape != pred_c2w.shape or gt_c2w.ndim != 3 or gt_c2w.shape[1:] != (4, 4):
        raise ValueError(
            f"expected matching Nx4x4 trajectories, got {gt_c2w.shape} and {pred_c2w.shape}"
        )
    if len(gt_c2w) < 2:
        raise ValueError("trajectory evaluation requires at least two poses")
    if not np.isfinite(gt_c2w).all() or not np.isfinite(pred_c2w).all():
        raise ValueError("trajectory contains non-finite values")

    # Sintel .cam rotations are stored with small numerical orthogonality
    # errors (up to roughly 2e-6 in the official evaluation subset). evo's
    # SO(3) logarithm rejects those matrices, so project both trajectories to
    # the nearest proper rotations before applying the standard protocol.
    gt_c2w, gt_rotation_projection = _project_rotations_to_so3(gt_c2w)
    pred_c2w, pred_rotation_projection = _project_rotations_to_so3(pred_c2w)

    reference = PosePath3D(poses_se3=list(gt_c2w))
    estimate = PosePath3D(poses_se3=list(pred_c2w))
    estimate_aligned = copy.deepcopy(estimate)
    align_rotation, align_translation, align_scale = estimate_aligned.align(
        reference, correct_scale=True
    )

    ate_metric = metrics.APE(PoseRelation.translation_part)
    ate_metric.process_data((reference, estimate_aligned))

    rpe_trans_metric = metrics.RPE(
        PoseRelation.translation_part,
        delta=1,
        delta_unit=Unit.frames,
        all_pairs=True,
    )
    rpe_trans_metric.process_data((reference, estimate_aligned))

    rpe_rot_metric = metrics.RPE(
        PoseRelation.rotation_angle_deg,
        delta=1,
        delta_unit=Unit.frames,
        all_pairs=True,
    )
    rpe_rot_metric.process_data((reference, estimate_aligned))

    return {
        "ate": float(ate_metric.get_statistic(StatisticsType.rmse)),
        "rpe_trans": float(rpe_trans_metric.get_statistic(StatisticsType.rmse)),
        "rpe_rot_deg": float(rpe_rot_metric.get_statistic(StatisticsType.rmse)),
        "align_scale": float(align_scale),
        "align_rotation": np.asarray(align_rotation).tolist(),
        "align_translation": np.asarray(align_translation).tolist(),
        "gt_rotation_projection_max_fro": gt_rotation_projection,
        "pred_rotation_projection_max_fro": pred_rotation_projection,
        "pred_c2w_aligned": np.asarray(estimate_aligned.poses_se3),
    }
