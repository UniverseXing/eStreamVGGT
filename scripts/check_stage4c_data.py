#!/usr/bin/env python3
"""Check frozen Stage 4C raw TUM RGB-D sequence coverage."""

import argparse
import os


SEQUENCES = (
    "rgbd_dataset_freiburg1_room",
    "rgbd_dataset_freiburg2_desk",
    "rgbd_dataset_freiburg3_long_office_household",
)


def read_timestamps(path):
    values = []
    with open(path) as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                values.append(float(stripped.split()[0]))
    return values


def association_count(root, sequence, max_difference):
    sequence_root = os.path.join(root, sequence)
    rgb_path = os.path.join(sequence_root, "rgb.txt")
    gt_path = os.path.join(sequence_root, "groundtruth.txt")
    if not os.path.isfile(rgb_path) or not os.path.isfile(gt_path):
        raise FileNotFoundError(
            f"{sequence}: expected rgb.txt and groundtruth.txt"
        )
    rgb = read_timestamps(rgb_path)
    gt = read_timestamps(gt_path)
    count = 0
    gt_index = 0
    for timestamp in rgb:
        while (
            gt_index + 1 < len(gt)
            and abs(gt[gt_index + 1] - timestamp)
            <= abs(gt[gt_index] - timestamp)
        ):
            gt_index += 1
        if abs(gt[gt_index] - timestamp) <= max_difference:
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
