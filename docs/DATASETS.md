# Evaluation datasets

Evaluation data is not redistributed by this repository. Obtain every dataset
under its own license and terms, then arrange it under `data/eval/` or use the
root overrides documented in
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

The preparation conventions follow the inherited StreamVGGT evaluation code
and, where applicable, the
[MonST3R evaluation guide](https://github.com/Junyi42/monst3r/blob/main/data/evaluation_script.md)
and [Spann3R preprocessing guide](https://github.com/HengyiWang/spann3r/blob/main/docs/data_preprocess.md).

## Dataset/task matrix

| Task | Dataset key | Default location |
|---|---|---|
| VideoDepth | `sintel` | `data/eval/sintel` |
| VideoDepth | `bonn` | `data/eval/bonn` |
| VideoDepth | `kitti` | `data/eval/kitti` |
| Camera pose | `sintel` | `data/eval/sintel/training/final` |
| Camera pose | `scannet` | `data/eval/scannetv2` |
| Camera pose | `tum` | `data/eval/tum` |
| Static reconstruction | `7scenes` | `data/eval/7scenes` |
| Static reconstruction | `nrgbd` | `data/eval/neural_rgbd` |
| Static reconstruction | `eth3d` | `data/eval/eth3d` |
| Dynamic reconstruction | `tum` | `data/eval/tum` |
| Long-sequence streaming | raw TUM | `data/eval/stage4c_tum` |

## Expected layout

The following tree shows the files read directly by the frozen evaluators.
Ellipses represent additional scenes or frames.

```text
data/eval/
├── sintel/training/
│   ├── final/<sequence>/*.png
│   ├── depth/<sequence>/*.dpt
│   └── camdata_left/<sequence>/*.cam
├── bonn/rgbd_bonn_dataset/
│   └── rgbd_bonn_<sequence>/
│       ├── rgb/*.png
│       ├── depth/*.png
│       └── groundtruth_110.txt
├── kitti/depth_selection/val_selection_cropped/
│   ├── image_gathered/<drive>/*.png
│   ├── groundtruth_depth_gathered/<drive>/*.png
│   └── stage4a_manifest.json
├── scannetv2/<scene>/
│   ├── color_90/*.jpg
│   └── pose_90.txt
├── tum/<sequence>/
│   ├── rgb_90/*
│   ├── groundtruth_90.txt
│   ├── rgb.txt
│   ├── depth.txt
│   └── depth/*
├── 7scenes/<scene>/
│   ├── TestSplit.txt
│   └── seq-XX/
│       ├── frame-XXXXXX.color.png
│       ├── frame-XXXXXX.depth.png
│       ├── frame-XXXXXX.depth.proj.png
│       └── frame-XXXXXX.pose.txt
├── neural_rgbd/<scene>/
│   ├── images/*
│   ├── depth/*
│   └── poses.txt
├── eth3d/<scene>/
│   ├── dslr_calibration_jpg/
│   │   ├── cameras.txt
│   │   └── images.txt
│   ├── images/dslr_images/*
│   └── ground_truth_depth/dslr_images/*
└── stage4c_tum/<sequence>/
    ├── rgb/*
    ├── rgb.txt
    └── groundtruth.txt
```

## KITTI VideoDepth

The repository includes a resumable downloader for the annotated validation
depths and the RGB data for the 13 drives in the frozen protocol:

```bash
bash scripts/reproduce/download_kitti.sh
```

By default, this downloads, prepares, and then validates all RGB/depth pairs.
An independent validation rerun is available as:

```bash
python scripts/check_stage4a_kitti.py --root data/eval/kitti
```

Preparation selects the first 110 paired frames per drive (or every available
pair for the four shorter drives) and uses
hardlinks when possible. Override the materialization mode when the source and
target are on different filesystems:

```bash
MATERIALIZE_MODE=symlink bash scripts/reproduce/download_kitti.sh
```

The public downloader accepts `KITTI_ROOT`, `PREPARE=0|1`, `VALIDATE=0|1`, and
`MATERIALIZE_MODE` (`hardlink`, `symlink`, or `copy`). `FRAMES_PER_DRIVE` is
frozen at `110`. Both preparation and validation default to enabled. Use
`DRY_RUN=1` to inspect its download, preparation, and validation commands
without changing the filesystem.

The downloader retains archives under `data/eval/kitti/downloads/` so an
interrupted download can resume. Remove them manually only after validating the
prepared data and accounting for your storage policy.

## Held-out raw TUM long sequences

Download the three frozen long-sequence inputs with:

```bash
bash scripts/reproduce/download_tum_long.sh
```

The wrapper validates at least 1000 associated RGB/ground-truth frames per
sequence by default. To rerun that check independently:

```bash
python scripts/check_stage4c_data.py \
  --root data/eval/stage4c_tum \
  --min-frames 1000
```

Only RGB images, timestamps, and motion-capture ground truth are extracted;
depth images are not used by this long-sequence evaluator. To remove the
downloaded archives after successful extraction:

```bash
DELETE_ARCHIVES=1 bash scripts/reproduce/download_tum_long.sh
```

Override the destination with `LONG_DATA_ROOT`. Validation is controlled by
`VALIDATE=0|1`, and its default threshold can be changed with `MIN_FRAMES`
(default `1000`). Set `DRY_RUN=1` to inspect the download and validation
commands.

The frozen sequences are:

- `rgbd_dataset_freiburg1_room`
- `rgbd_dataset_freiburg2_desk`
- `rgbd_dataset_freiburg3_long_office_household`

Keep this raw held-out root separate from the MonST3R-prepared TUM root used by
pose and dynamic reconstruction.

## 7-Scenes projected depth

After downloading and extracting the seven official scenes, register the raw
depth maps to the RGB camera:

```bash
python scripts/prepare_stage3_3b_7scenes.py \
  --root data/eval/7scenes \
  --workers 7
```

This produces `frame-XXXXXX.depth.proj.png` without requiring GUI OpenCV. It is
safe to rerun: existing projected maps are retained unless `--overwrite` is
specified.

Once 7-Scenes, NRGBD, and ETH3D are all present, validate their joint layout:

```bash
python scripts/check_stage3_3b_data.py --root data/eval
```

For ETH3D, `cameras.txt` and `images.txt` must be directly under each scene's
`dslr_calibration_jpg/`; an additional archive-created nesting level is not
accepted by the evaluator.

## TUM Dynamics

The dynamic reconstruction protocol expects the eight Freiburg 3
sitting/walking sequences prepared with aligned `rgb_90` images and
`groundtruth_90.txt`. It also reads the original `rgb.txt`, `depth.txt`, and
`depth/` directory to associate depth frames.

Validate the complete frozen set with:

```bash
python scripts/check_stage3_3c_data.py \
  --root data/eval/tum \
  --frames 50
```

The `--allow-subset` option is useful only for a local smoke test; do not use a
subset when comparing against the frozen aggregate table.

## Sintel, Bonn, and ScanNet

Use the inherited MonST3R preparation conventions:

- Sintel needs the `final`, `depth`, and `camdata_left` training archives with
  matching sequence/frame names.
- Bonn needs RGB, depth, and the aligned `groundtruth_110.txt` within each
  `rgbd_bonn_<sequence>` directory.
- ScanNet pose evaluation expects sampled RGB images in `color_90/` and a
  matching finite camera trajectory in `pose_90.txt`.

Do not silently change sampling, frame order, crop, or alignment files when
comparing with the provided tables.

## NRGBD and ETH3D

NRGBD scenes require `images/`, `depth/`, and `poses.txt`. ETH3D scenes require
COLMAP-format DSLR calibration and poses, DSLR RGB images, and matching raw
ground-truth depth maps in the layout shown above.

The paper/dense reconstruction protocols choose frames differently. Use the
public reconstruction wrapper and record the selected protocol rather than
manually choosing favourable views.

## Data integrity and version control

Datasets and generated predictions should remain outside Git. Before a formal
run:

1. run the applicable checker;
2. record the dataset root and protocol in the job log;
3. keep the same data preparation for every compared method;
4. verify that every method evaluates the same eligible sequence set; and
5. retain the evaluator JSON/CSV metadata needed to audit coverage and failures.
