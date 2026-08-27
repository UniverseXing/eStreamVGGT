# Algorithm S1: coupled bounded historical-state update

## Descriptor and cached state

For each incoming frame, the frozen DINOv2 ViT-L/14 register backbone produces
final normalised patch tokens of dimension 1024. The selector descriptor is the
L2-normalised mean of those patch tokens. It is detached from autograd and
stored in float32, hence it occupies exactly 4096 bytes per retained frame.
No second image encoder or re-encoding pass is introduced.

Each retained frame ID refers jointly to:

- one frame item on axis 2 of every aggregator K/V tensor;
- four frame items on axis 2 of every camera-head K/V tensor; and
- one 1024-dimensional DINO descriptor when the policy uses DINO.

The same selected frame indices are applied to both KV branches. RGB images and
dense predictions are not part of the cache state.

## Shared online update

```text
Input: previous coupled states, incoming frame t, budget K, frozen policy P

1. Encode frame t once and append its aggregator KV and descriptor.
2. Form candidate IDs by appending t to the retained historical IDs.
3. If the candidate count exceeds K (or K8 bank assignment is active), compute
   keep_indices according to P. Stable sorting and increasing frame ID resolve
   equal-score ties deterministically.
4. Slice every aggregator K and V tensor on frame axis 2 using keep_indices.
5. Run the camera head for frame t, then expand each frame index to its four
   camera items and slice every camera-head K and V tensor on axis 2.
6. Slice the descriptor array and frame-ID array using the same keep_indices.
7. Emit the current prediction. With streaming release, write it to the output
   sink and release its GPU tensors before frame t+1.
```

Deletion therefore happens after the incoming aggregator state has become a
candidate and after the current representation has queried the previous
history. The camera and aggregator caches cannot silently retain different
frame identities in the paper configurations.

## K4

Warm-up retains every candidate until four states are available. In steady
state the five candidates comprise the persistent frame-0 anchor, three
replaceable historical candidates, and the current frame. The selector keeps:

```text
frame 0 anchor
two old candidates with the smallest maximum cosine similarity to the current frame
current frame
```

The DINO score is computed by a matrix product between L2-normalised
descriptors, so it is cosine similarity. Lower maximum similarity means less
visual redundancy. Stable score ordering followed by increasing temporal order
makes ties deterministic.

## K6

Warm-up again retains all candidates. In steady state the seven candidates are
split into:

```text
frame 0 anchor
two DINO-selected old candidates
the three most recent candidates, including current
```

Each old candidate is scored by its maximum cosine similarity to the recent
set. The two lowest-scoring old candidates are retained, then all selected IDs
are restored to temporal order before tensor slicing.

## K8

K8 uses fixed age banks relative to current frame ID t:

```text
anchor: frame ID 0
long:   age >= 48, one DINO-diverse landmark
middle: age 16--47, the oldest available candidate
near:   age 4--15, the oldest available candidate
recent: age 0--3, up to four frames including current
```

The long-bank candidate having the lowest maximum cosine similarity to the
already selected anchor/recent reference set is retained. Middle and near use
the oldest candidate so that a landmark is guaranteed to age into the next
bank. Empty banks are skipped during warm-up; unused capacity is filled with
the most recent unselected candidates. Once all banks are populated, the
steady-state layout is exactly `1 anchor + 3 landmarks + 4 recent = 8`.

All age bounds are inclusive as written. Frame IDs are zero-based internally;
paper figures may display one-based view numbers but must label that conversion.
