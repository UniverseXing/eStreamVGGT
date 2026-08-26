# Single-scene qualitative comparison

This workflow runs the same existing dataset sequence with Full cache, K4, K6,
and K8, then exports 12 independent PNG panels. It does not capture a camera
or automatically assemble the final paper figure.

## Frozen scene

The recommended scene is Bonn `person_tracking2`, using its complete 110-frame
sampled RGB sequence and displaying frame 110. It contains appreciable camera
and human motion, all four methods can complete the same input on the RTX 6000
Ada server, and the chosen frame is the fixed sequence endpoint rather than a
post-hoc attractive frame.

Expected input directory:

```text
data/eval/bonn/rgbd_bonn_dataset/rgbd_bonn_person_tracking2/rgb_110_sampled
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
  --images-dir data/eval/bonn/rgbd_bonn_dataset/rgbd_bonn_person_tracking2/rgb_110_sampled \
  --sequence person_tracking2 \
  --max-frames 110 \
  --frame 110 \
  --output-dir paper_assets/qualitative/bonn_person_tracking2_f110
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

Every image is 1200 by 900 pixels and contains the method, sequence, and
`frame 110/110` label. RGB is intentionally repeated because all methods receive
the same selected input frame.

The directory also contains `qualitative_metadata.json` and
`qualitative_caption.txt`. The caption records the detected GPU, online input
and output lifecycle, shared visualisation settings, and the absence of bundle
adjustment, ICP, temporal smoothing, or geometric refinement.

Point confidence filtering, uniform point subsampling, a rigid x/z/-y display
axis permutation, and percentile clipping for depth display are used only for
legible rendering and are disclosed in the generated caption.
