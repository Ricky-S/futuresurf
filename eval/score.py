"""FutureSurf gap scorer. Bidirectional Chamfer (mean nearest-neighbor distance) between
mesh vertex sets; observed/future split at frame 150; gap = future/observed CD."""
import argparse
import glob
import json
import os
import re

import numpy as np
import trimesh
from scipy.spatial import cKDTree

SPLIT_FRAME, N_FRAMES = 150, 200


def trailing_int(p):
    n = re.findall(r'(\d+)', p.rsplit('/', 1)[-1])
    return int(n[-1]) if n else -1


def chamfer(a, b):
    d = cKDTree(b).query(a)[0]
    return float(np.mean(d))


def eval_one(gt_obj, pred_ply):
    gt_v = np.asarray(trimesh.load(gt_obj, force='mesh').vertices, float)
    pr_v = np.asarray(trimesh.load(pred_ply, force='mesh').vertices, float)
    return (chamfer(gt_v, pr_v) + chamfer(pr_v, gt_v)) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', required=True)
    ap.add_argument('--pred_dir', required=True)
    ap.add_argument('--gt_dir', required=True)
    ap.add_argument('--out', default='.')
    ap.add_argument('--split_frame', type=int, default=SPLIT_FRAME)
    a = ap.parse_args()

    preds = {trailing_int(p): p for p in glob.glob(a.pred_dir + '/frame_*.ply')}
    gt_objs = sorted(glob.glob(a.gt_dir + '/*.obj') + glob.glob(a.gt_dir + '/*.ply'), key=trailing_int)

    obs, fut, missing = [], [], 0
    for i, gt in enumerate(gt_objs[:N_FRAMES]):
        if i not in preds:
            missing += 1
            continue
        cd = eval_one(gt, preds[i])
        (obs if i < a.split_frame else fut).append(cd)

    o, f = np.array(obs), np.array(fut)
    res = dict(scene=a.scene, n_observed=len(o), n_future=len(f), n_missing_pred=missing,
               observed_mean_cd=float(o.mean()), future_mean_cd=float(f.mean()),
               gap_ratio_mean=float(f.mean() / o.mean()), gap_ratio_median=float(np.median(f) / np.median(o)))

    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, f'gap_{a.scene}.json'), 'w'), indent=2)
    print(f"[CPU] {a.scene}: obs={res['observed_mean_cd']:.6f} fut={res['future_mean_cd']:.6f} "
          f"GAP={res['gap_ratio_mean']:.4f} (median {res['gap_ratio_median']:.4f})")


if __name__ == '__main__':
    main()
