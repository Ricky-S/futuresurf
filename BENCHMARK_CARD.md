# Benchmark Card: FutureSurf (constructed motions)

Future-time surface reconstruction: train on the observed window, score the extracted **mesh** on the held-out future. Companion to the paper "Future Rendering ≠ Future Surface".

## Intended use
Compare methods' relative future-surface accuracy under one method-agnostic split: train on `time ≤ 0.75`,
score extracted meshes at `time > 0.75` by per-frame symmetric Chamfer in the canonical frame.

## Supported claims
- **Relative** future-surface accuracy across methods (same split, same scoring).
- **Motion-type dependence** of extrapolation difficulty (via the eight motions + controls).
- **Rendering ≠ surface**: NVS metrics (PSNR/SSIM/LPIPS) do not predict future-surface accuracy.

## Unsupported claims
- Real-world performance (data is synthetic).
- Within-window quality (this measures extrapolation, not interpolation).
- Appearance / NVS quality (not scored).
- Cross-scene aggregation of absolute CD (scales are per-scene; compare per scene).

## Assumptions
- **Per-frame GT meshes** at every evaluated time (unobtainable for real captures).
- **Monocular orbit** capture.
- **Shared canonical frame**: predictions and GT are in the same frame (Y-up, origin-centered); the scorer
  does no alignment (no per-frame fit, no ICP/Sim(3)).

## Metric definitions
- **Per-frame CD:** symmetric (bidirectional) Chamfer between the extracted and GT mesh vertices in the
  canonical frame. With `p` over the predicted vertices and `g` over the GT vertices:
  `½·( mean_p min_g ‖p−g‖ + mean_g min_p ‖g−p‖ )` — the average distance from each vertex to the nearest
  vertex of the other mesh, taken both ways (the surface-reconstruction convention, e.g. DTU); matches
  `score.py`. Frames are rank-aligned by trailing index.
- **Primary score:** absolute `future_mean_cd` per scene (GT scale, run-stable), from `eval/score.py`.
- **Gap:** `future_mean_cd / observed_mean_cd` (observed `0..149`, future `150..199`). `1` = the future
  surface is as accurate as the observed window; `> 1` = worse.
- **GT-side oracle (`eval/gt_oracle.py`):** simple extrapolators on the GT trajectories, bounding how much
  of the future is recoverable from observed motion.

## Controls (internal validity)
- **`twist`** (surface-invariant): vertices move but the surface barely changes → `≈ 1×` gap. Excluded from gap stats.
- **`rotate`** (pure gauge): rigid rotation; error almost fully Sim(3)-removable.
- **`stop`** (static future): motion freezes before the split; the metric exposes drift from "nothing moves".

## Known limitations
- **Scale:** 8 motions of a single textured-sphere base.
- **Synthetic constraint:** future-surface GT needs animated assets, so results characterize the task +
  backbone inductive bias, not real-capture performance.
- Absolute CDs are per-scene-scale; only the gap ratio compares across scenes.

## Provenance & licensing
- **Constructed motions (8):** our Blender-EEVEE renders + analytic per-frame meshes (construction in the
  paper). **CC BY 4.0**.
- **Evaluation code (`eval/`):** **MIT** (`LICENSE`).
