#!/usr/bin/env python3
"""Check frozen Stage 4C raw TUM RGB-D sequence coverage."""

import argparse
import math
import os


SEQUENCES = (
    "rgbd_dataset_freiburg1_room",
    "rgbd_dataset_freiburg2_desk",
    "rgbd_dataset_freiburg3_long_office_household",
)


def read_rows(path):
    rows = []
    with open(path) as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                fields = stripped.split()
                rows.append((float(fields[0]), fields[1:]))
    if not rows:
        raise ValueError(f"no records in {path}")
    return rows


def association_count(root, sequence, max_difference):
    sequence_root = os.path.join(root, sequence)
    rgb_path = os.path.join(sequence_root, "rgb.txt")
    gt_path = os.path.join(sequence_root, "groundtruth.txt")
    if not os.path.isfile(rgb_path) or not os.path.isfile(gt_path):
        raise FileNotFoundError(
            f"{sequence}: expected rgb.txt and groundtruth.txt"
        )
    rgb = read_rows(rgb_path)
    gt = read_rows(gt_path)
    gt_timestamps = [row[0] for row in gt]
    count = 0
    gt_index = 0
    for timestamp, rgb_fields in rgb:
        if not rgb_fields:
            raise ValueError(f"{sequence}: RGB row has no image path")
        while (
            gt_index + 1 < len(gt)
            and abs(gt_timestamps[gt_index + 1] - timestamp)
            <= abs(gt_timestamps[gt_index] - timestamp)
        ):
            gt_index += 1
        if abs(gt_timestamps[gt_index] - timestamp) <= max_difference:
            pose_fields = gt[gt_index][1]
            if len(pose_fields) != 7:
                raise ValueError(
                    f"{sequence}: expected 7 pose values, found {len(pose_fields)}"
                )
            pose = [float(value) for value in pose_fields]
            if not all(math.isfinite(value) for value in pose):
                raise ValueError(f"{sequence}: non-finite ground-truth pose")
            image_path = os.path.join(sequence_root, rgb_fields[0])
            if not os.path.isfile(image_path):
                raise FileNotFoundError(
                    f"{sequence}: missing associated RGB image {image_path}"
                )
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser("Check Stage 4C TUM data")
    parser.add_argument("--root", default="data/eval/stage4c_tum")
    parser.add_argument("--sequences", nargs="+", default=SEQUENCES)
    parser.add_argument("--min-frames", type=int, default=1000)
    parser.add_argument("--max-association-difference", type=float, default=0.02)
    args = parser.parse_args()

    for sequence in args.sequences:
        count = association_count(
            args.root, sequence, args.max_association_difference
        )
        if count < args.min_frames:
            raise ValueError(
                f"{sequence}: {count} associated frames, "
                f"need {args.min_frames}"
            )
        print(f"{sequence}: {count} associated RGB/GT frames")
    print("Stage 4C data check: PASS")


if __name__ == "__main__":
    main()
