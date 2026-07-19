"""FutureSurf GT-side recoverability oracle. Fits simple extrapolators (freeze / velocity /
acceleration / phase-wrap + ridge poly/Fourier) on the ground-truth vertex trajectories: fit on the
observed window, predict the held-out future, score vs future GT. Bounds how recoverable each
motion's future is from observed motion alone. Metric: bbox-normalized linear bidirectional Chamfer.

Usage:
  python gt_oracle.py --data dataset/controlled --out gtside_constructed.json
"""
import argparse
import glob
import json
import re
import sys

import numpy as np
import trimesh
from scipy.spatial import cKDTree

NF = 200
SPLIT = 150
FIT_END = 120
T = float(NF)
LAM = 1e-3
N_FUT = 12
N_VAL = 8
RECOVER_THR = 0.005
LEARNED = [('poly1', 1), ('poly2', 2), ('poly3', 3),
           ('fourierK2', -2), ('fourierK4', -4), ('fourierK6', -6)]
NAIVE = ['freeze', 'velocity', 'acceleration', 'phasewrap']


def ti(p):
    n = re.findall(r'(\d+)', p.rsplit('/', 1)[-1])
    return int(n[-1]) if n else -1


def cd(a, b):
    da = cKDTree(b).query(a)[0]
    db = cKDTree(a).query(b)[0]
    return (da.mean() + db.mean()) / 2 / (np.linalg.norm(b.max(0) - b.min(0)) + 1e-9)


def feats(ts, code):
    ts = np.asarray(ts, float)
    if code > 0:
        return np.stack([ts ** d for d in range(code + 1)], 1)
    K = -code
    cols = [np.ones_like(ts), ts]
    for k in range(1, K + 1):
        w = 2 * np.pi * k / T
        cols += [np.sin(w * ts), np.cos(w * ts)]
    return np.stack(cols, 1)


def learned_predict(fit_t, fit_X, pred_t, code):
    Phi = feats(fit_t, code)
    A = Phi.T @ Phi + LAM * np.eye(Phi.shape[1])
    W = np.linalg.solve(A, Phi.T @ fit_X.reshape(len(fit_t), -1))
    return (feats(pred_t, code) @ W).reshape(len(pred_t), fit_X.shape[1], 3)


def detect_period(Vd, fit_i):
    base = Vd[fit_i[-1]]
    return max(range(5, min(120, len(fit_i))),
               key=lambda L: -np.mean(np.linalg.norm(base - Vd[fit_i[-1 - L]], axis=1))
               if (len(fit_i) - 1 - L) >= 0 else -1e9)


def naive_predict(Vd, fit_i, pred_i, method):
    a, b, c = fit_i[-3], fit_i[-2], fit_i[-1]
    last = Vd[c]
    vel = Vd[c] - Vd[b]
    acc = Vd[c] - 2 * Vd[b] + Vd[a]
    per = detect_period(Vd, fit_i) if method == 'phasewrap' else 1
    split_local = c + 1
    out = []
    for i in pred_i:
        dt = i - c
        if method == 'freeze':
            out.append(last)
        elif method == 'velocity':
            out.append(last + vel * dt)
        elif method == 'acceleration':
            out.append(last + vel * dt + 0.5 * acc * dt * dt)
        else:
            src_frame = (split_local - per) + ((i - split_local) % per)
            out.append(Vd.get(src_frame, last))
    return out


def load_scene(data, scene):
    gt_dir = f'{data}/{scene}/mesh_gt'
    gts = {ti(p): p for p in glob.glob(gt_dir + '/*.obj') + glob.glob(gt_dir + '/*.ply')}
    if (SPLIT - 1) not in gts:
        return None, None
    nV = len(trimesh.load(gts[SPLIT - 1], force='mesh').vertices)
    have = [i for i in range(NF) if i in gts
            and len(trimesh.load(gts[i], force='mesh').vertices) == nV]
    Vd = {i: np.asarray(trimesh.load(gts[i], force='mesh').vertices, float) for i in have}
    return Vd, have


def _sample(idx, n):
    return [idx[j] for j in np.linspace(0, len(idx) - 1, min(n, len(idx))).astype(int)]


def per_rule_detail(Vd, have):
    obs_i = [i for i in have if i < SPLIT]
    fut_all = [i for i in have if i >= SPLIT]
    fut_s = _sample(fut_all, N_FUT)
    obs_X = np.stack([Vd[i] for i in obs_i])

    out = {}
    for m in NAIVE:
        preds = naive_predict(Vd, obs_i, fut_s, m)
        out[f'naive_{m}'] = float(np.mean([cd(p, Vd[fut_s[j]]) for j, p in enumerate(preds)]))
    for name, code in LEARNED:
        pred = learned_predict(np.array(obs_i, float), obs_X, np.array(fut_s, float), code)
        out[f'learned_{name}'] = float(np.mean([cd(pred[j], Vd[fut_s[j]]) for j in range(len(fut_s))]))
    out['ref_freeze'] = out['naive_freeze']
    out['ref_frame_jitter'] = float(cd(Vd[fut_s[0]], Vd[min(fut_s[0] + 1, max(have))]))
    return out


def run_scene(Vd, have):
    fit_i = [i for i in have if i < FIT_END]
    val_i = [i for i in have if FIT_END <= i < SPLIT]
    obs_i = [i for i in have if i < SPLIT]
    fut_all = [i for i in have if i >= SPLIT]
    if len(fit_i) < 20 or len(val_i) < 5 or not fut_all:
        return None
    val_s = _sample(val_i, N_VAL)
    fut_s = _sample(fut_all, N_FUT)
    fit_X = np.stack([Vd[i] for i in fit_i])
    obs_X = np.stack([Vd[i] for i in obs_i])

    def fut_cd_learned(code, fit_t, fit_arr):
        pred = learned_predict(np.array(fit_t, float), fit_arr, np.array(fut_s, float), code)
        return np.mean([cd(pred[j], Vd[fut_s[j]]) for j in range(len(fut_s))])

    lv = {name: np.mean([cd(learned_predict(np.array(fit_i, float), fit_X, np.array([v], float), code)[0], Vd[v])
                         for v in val_s]) for name, code in LEARNED}
    nv = {m: np.mean([cd(p, Vd[v]) for p, v in zip(naive_predict(Vd, fit_i, val_s, m), val_s)]) for m in NAIVE}
    sel_L = min(lv, key=lv.get)
    sel_N = min(nv, key=nv.get)
    code_sel = dict(LEARNED)[sel_L]
    L_fut = fut_cd_learned(code_sel, obs_i, obs_X)
    N_fut = np.mean([cd(p, Vd[fut_s[j]]) for j, p in enumerate(naive_predict(Vd, obs_i, fut_s, sel_N))])
    L_oracle = min(fut_cd_learned(code, obs_i, obs_X) for _, code in LEARNED)
    N_oracle = min(np.mean([cd(p, Vd[fut_s[j]]) for j, p in enumerate(naive_predict(Vd, obs_i, fut_s, m))])
                   for m in NAIVE)
    return dict(sel_learned=sel_L, sel_naive=sel_N, learned_fut=float(L_fut), naive_fut=float(N_fut),
                learned_oracle=float(L_oracle), naive_oracle=float(N_oracle),
                learned_beats_naive_nopeek=bool(L_fut < N_fut))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, help='scene root containing <scene>/mesh_gt/')
    ap.add_argument('--scenes', default=None, help='comma-separated; default = the 8 constructed motions')
    ap.add_argument('--out', default=None, help='output json path')
    ap.add_argument('--validate', default=None, help='reference json to compare against (max abs cell diff)')
    ap.add_argument('--tol', type=float, default=1e-9)
    args = ap.parse_args()

    scenes = args.scenes.split(',') if args.scenes else \
        ['wave', 'compound', 'stretch', 'bulge', 'accel', 'twist', 'rotate', 'stop']

    res = {}
    for s in scenes:
        print(f'=== {s} ===', flush=True)
        Vd, have = load_scene(args.data, s)
        if Vd is None:
            print(f'  [skip] no topology-consistent GT at {args.data}/{s}/mesh_gt')
            continue
        detail = per_rule_detail(Vd, have)
        summary = run_scene(Vd, have)
        rule_keys = [k for k in detail if k.startswith(('naive_', 'learned_'))]
        best_name = min(rule_keys, key=lambda k: detail[k])
        best_val = detail[best_name]
        res[s] = dict(summary=summary, per_rule=detail, best_rule=best_name,
                      best_rule_fut_cd=best_val, recovered=int(best_val < RECOVER_THR))
        print(f'  best rule: {best_name} fut CD={best_val:.5f}  '
              f'freeze={detail["naive_freeze"]:.5f}  recovered={res[s]["recovered"]}')

    if args.out:
        json.dump(res, open(args.out, 'w'), indent=2)
        print(f'\nwrote {args.out}')

    if args.validate:
        ref = json.load(open(args.validate))
        maxd, worst = 0.0, None
        for s in res:
            cells = res[s]['per_rule']
            ref_cells = ref[s]['per_rule']
            for k, v in cells.items():
                if isinstance(v, float) and k in ref_cells:
                    d = abs(v - ref_cells[k])
                    if d > maxd:
                        maxd, worst = d, (s, k, ref_cells[k], v)
        print(f'\nvalidate vs {args.validate}: max abs cell diff = {maxd:.3e}')
        if worst:
            print(f'  worst cell: scene={worst[0]} rule={worst[1]} ref={worst[2]:.7f} got={worst[3]:.7f}')
        sys.exit(0 if maxd <= args.tol else 1)


if __name__ == '__main__':
    main()
