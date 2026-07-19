# FutureSurf: controlled motions

Eight **controlled motions** of a deforming sphere (wave, compound, stretch, bulge, accel, plus the twist/rotate/stop controls), part of the **FutureSurf** benchmark: train on the observed window, then predict the *future* surface.

## Layout (per motion)

```
controlled/<motion>/
  images/                 r_0.png … r_199.png       200 renders, 800×800 RGBA
  mesh_gt/                <motion>0.ply … 199.ply    200 GT meshes (vertex-colored PLY)
  transforms_train.json   observed: 150 frames (time ≤ 0.75)
  transforms_test.json    future:    50 frames (time > 0.75)
  points3d.ply            init point cloud for Gaussian backbones
```

## Use it

Train on `transforms_train.json` (the observed 75%), then score the extracted mesh against `mesh_gt` at the future frames. The metric is Chamfer on vertex **positions**; the per-vertex color and normals in the PLY are for visualization only.
