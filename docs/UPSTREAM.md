# Relationship to StreamVGGT

eStreamVGGT is an independent research extension of
[StreamVGGT](https://github.com/wzzheng/StreamVGGT), not an official fork or
release maintained by the original StreamVGGT authors.

## Base provenance

The current eStreamVGGT history is based on upstream StreamVGGT commit
[`05682bcb`](https://github.com/wzzheng/StreamVGGT/commit/05682bcb05683c4c71522e9cb618fc8ca349063c).
The Git remotes used during development are:

```text
origin   https://github.com/UniverseXing/eStreamVGGT.git
upstream https://github.com/wzzheng/StreamVGGT.git
```

The original repository provides the StreamVGGT architecture, model training,
checkpoint, demo, and its base evaluation stack. Its public resources are:

- [Paper: Streaming 4D Visual Geometry Transformer](https://arxiv.org/abs/2507.11539)
- [Project page](https://wzzheng.net/StreamVGGT)
- [Source repository](https://github.com/wzzheng/StreamVGGT)
- [Checkpoint repository](https://huggingface.co/lch01/StreamVGGT/)
- [Online demo](https://huggingface.co/spaces/lch01/StreamVGGT)

## eStreamVGGT changes

This repository adds or substantially extends:

- frame-level cache-budget enforcement in `StreamVGGT.inference`;
- DINO-guided K4 and K6 history selection;
- temporally banked DINO-guided K8 selection;
- coupled aggregator/camera-head cache trimming;
- cache selection and memory trace logging;
- streaming input and output-sink release for long sequences;
- robust trajectory metrics and task-specific failure records;
- VideoDepth, pose, reconstruction, and held-out long-sequence protocols;
- same-GPU resource/quality audits and frozen decision gates; and
- deterministic paper/supplementary asset builders with source hashes.

The bounded methods change state management at inference time. They use the
original StreamVGGT checkpoint and do not claim a separately trained model.

Historical stage-numbered scripts and legacy policy names are retained where
needed to audit the development process. The stable public interface is the
canonical method set and wrappers under `scripts/reproduce/`.

## What remains upstream

The inherited training, fine-tuning, FlashAttention integration, and Gradio
demo are not the primary contribution of eStreamVGGT. Refer to upstream for
their authoritative documentation and support.

Assets such as the original teaser/result images, inherited source modules,
and the StreamVGGT checkpoint must continue to be attributed to their original
authors. Dataset and third-party repository licenses are separate from the
repository license.

## Citation

The eStreamVGGT paper and its final BibTeX entry are coming soon. Until then,
do not invent or cite a placeholder publication. Work using this repository
should cite the original StreamVGGT paper:

```bibtex
@article{streamVGGT,
  title={Streaming 4D Visual Geometry Transformer},
  author={Dong Zhuo and Wenzhao Zheng and Jiahe Guo and Yuqi Wu and Jie Zhou and Jiwen Lu},
  journal={arXiv preprint arXiv:2507.11539},
  year={2025}
}
```

When the eStreamVGGT paper is released, cite both the extension paper and the
upstream StreamVGGT paper.

## License

The repository retains StreamVGGT's
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license](../LICENSE.txt).
Review that license before redistributing or adapting the code, and separately
review the terms for model weights, datasets, and incorporated third-party
components.

## Updating from upstream

Future upstream changes should be integrated explicitly rather than by
replacing this repository with a fresh clone. A maintainer can inspect the
difference with:

```bash
git fetch upstream
git log --oneline --left-right upstream/main...main
git diff upstream/main...main
```

After an upstream merge or rebase, rerun the smoke test and all affected
evaluation gates. Cache tensor layout, aggregator return values, and camera
head KV semantics are particularly sensitive integration points.
