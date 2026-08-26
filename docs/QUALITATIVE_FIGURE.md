# Single-scene qualitative comparison

This workflow runs the same existing dataset sequence with Full cache, K4, K6,
and K8, then exports 12 independent PNG panels. It does not capture a camera
or automatically assemble the final paper figure.

## Frozen object-centred scene

The recommended scene is 7-Scenes `chess/seq-01`. The camera moves around the
same chessboard through wide, close, overhead, and oblique views, making it more
suitable for showing multi-view object reconstruction than a person-tracking
video. The input is frozen as the first 110 images obtained with source-frame
stride 5: frames `000000, 000005, ..., 000545`. This gives 110 inputs while
covering substantially more viewpoint change than 110 consecutive video frames.
The displayed result is sampled view 110 (source frame `000545`), so no frame is
selected after inspecting model quality.

Expected input directory:

```text
data/eval/7scenes/chess/seq-01
```

## Command

On the cluster, submit the existing root `run.sh`. Its original Conda activation
is unchanged; `STREAMVGGT_RUN_TARGET` only selects this qualitative entry point:

```bash
STREAMVGGT_RUN_TARGET=qualitative \
sbatch \
  --job-name=streamvggt-qualitative \
  --output=streamvggt-qualitative-%j.out \
  --error=streamvggt-qualitative-%j.err \
  run.sh
```

The resolved defaults are equivalent to this direct command in an already
allocated and activated GPU shell:

```bash
python scripts/reproduce/run_qualitative_figure.py \
  --repo-root . \
  --weights ckpt/checkpoints.pth \
  --images-dir data/eval/7scenes/chess/seq-01 \
  --sequence chess_seq01 \
  --image-glob '*.color.png' \
  --sampling-stride 5 \
  --max-frames 110 \
  --frame 110 \
  --output-dir paper_assets/qualitative/7scenes_chess_seq01_v110
```

The script uses causal one-frame-at-a-time loading and the same CPU output sink
for every method. The four depth panels share one display range; the four point
cloud panels share one viewpoint and coordinate range.

## Twelve PNG panels

The columns should be assembled in this order:

```text
Full cache | K4 | K6 | K8
```

Each column contains:

```text
RGB input
Depth prediction
Point cloud / reconstruction
```

The exact output files are:

```text
full_cache_rgb.png
full_cache_depth.png
full_cache_pointcloud.png
k4_rgb.png
k4_depth.png
k4_pointcloud.png
k6_rgb.png
k6_depth.png
k6_pointcloud.png
k8_rgb.png
k8_depth.png
k8_pointcloud.png
```

Every image is 1200 by 900 pixels and contains the method, sequence, sampled-view
index, and original source-frame index. RGB is intentionally repeated because
all methods receive the same selected input view.

The directory also contains `qualitative_metadata.json` and
`qualitative_caption.txt`. The caption records the detected GPU, online input
and output lifecycle, shared visualisation settings, and the absence of bundle
adjustment, ICP, temporal smoothing, or geometric refinement.

Point confidence filtering, uniform point subsampling, a rigid x/z/-y display
axis permutation, and percentile clipping for depth display are used only for
legible rendering and are disclosed in the generated caption.
