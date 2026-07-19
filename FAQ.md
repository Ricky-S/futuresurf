# FutureSurf: FAQ

**Q: What is FutureSurf?**
Train on the first 75% of a scene's timestamps, then score the extracted *mesh* at the held-out future 25% against per-frame ground-truth meshes.

**Q: What's in the release?**
Eight analytic motions of a textured sphere (200 frames at 800×800 each, on a shared orbit), the exact GT mesh per frame, the 150/50 split files, and the scoring code (`score.py`, `gt_oracle.py`). Blender convention (`transforms_*.json` + images); meshes are PLY.

**Q: Why synthetic data?**
No sensor gives the *true future surface* of a real scene; analytic motions make the reference exact and the difficulty controllable.

**Q: How do I evaluate my own method?**
Meshes in → JSON out. Produce one mesh per frame (frames 0–199, in the dataset's canonical frame), run `eval/score.py` (~1-2 min/scene). See "Evaluate your method" in the README.

**Q: Which is the score — gap ratio or absolute future CD?**
Absolute future Chamfer distance is the primary score. The future/observed gap ratio is reported as a diagnostic: because it divides by observed-window CD, small run-to-run changes in the observed reconstruction error can change the ratio.
