"""Probe: a one-arm freeze on the FLAT edge of the left claw (bent arm), not on its tip.

The support is every claw vertex within 2 mm of the floor; its convex hull is the support area.
Wanted: an area at least 4 cm long, the CoM inside it with margin (then the pose is statically
stable, not balanced on a point), every other limb >= 3 cm off the floor, the body held high,
and small joint torques with the whole weight on the claw (centre of pressure under the CoM).
"""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

import freeze_pose as fp
from freeze_pose import m, d, B, mass, g, VERT, SUPPORT, OTHER, NAMES, qadr, dadr, rng
from tasks.jumper.common.actuator import PLATEAU_TORQUE

TIP = B('LF_finger_tip_link')
PALM = B('LF_palm_link')
# The claw's upper edge, as a direction in the palm's own frame: whatever points up when the
# robot stands at HOME. "Upper edge on the floor" = that direction pointing down in the freeze.
d.qpos[:] = 0; d.qpos[2] = fp.rs.STAND_Z; d.qpos[3] = 1
d.qpos[qadr] = [fp.HOME[n] for n in NAMES]; mujoco.mj_forward(m, d)
UP_LOCAL = d.xmat[PALM].reshape(3, 3).T @ np.array([0, 0, 1.0])
_tip_home = d.xpos[TIP] - d.xpos[PALM]
AXIS_LOCAL = d.xmat[PALM].reshape(3, 3).T @ (_tip_home / np.linalg.norm(_tip_home))   # palm -> finger tip


def place(x):
    d.qpos[:] = 0; d.qpos[2] = 1.0
    q = np.zeros(4); mujoco.mju_euler2Quat(q, np.array([x[0], x[1], 0.0]), 'xyz'); d.qpos[3:7] = q
    d.qpos[qadr] = np.clip(x[2:], rng[:, 0], rng[:, 1])
    mujoco.mj_forward(m, d)
    pts = {gi: VERT[gi] @ d.geom_xmat[gi].reshape(3, 3).T + d.geom_xpos[gi] for gi in SUPPORT}
    low = min(p[:, 2].min() for p in pts.values())
    d.qpos[2] -= low; mujoco.mj_forward(m, d)


def evaluate(x):
    place(x)
    pts = [(VERT[gi] @ d.geom_xmat[gi].reshape(3, 3).T + d.geom_xpos[gi], m.geom_bodyid[gi]) for gi in SUPPORT]
    patch = np.concatenate([p[p[:, 2] < 0.002] for p, b in pts])
    tip_only = all(b == TIP for p, b in pts if (p[:, 2] < 0.002).any())
    clearance = min(float((VERT[gi] @ d.geom_xmat[gi].reshape(3, 3).T + d.geom_xpos[gi])[:, 2].min()) for gi in OTHER)
    com = d.subtree_com[1].copy()
    xy = patch[:, :2]
    length = float(np.ptp(xy @ np.linalg.svd(xy - xy.mean(0))[2][0])) if len(xy) > 2 else 0.0
    try:
        hull = ConvexHull(xy)
        margin = float(-(hull.equations[:, :2] @ com[:2] + hull.equations[:, 2]).max())
    except Exception:
        margin = -float(np.min(np.linalg.norm(xy - com[:2], axis=1))) if len(xy) else -1.0
    cop = np.array([com[0], com[1], 0.0])
    jac = np.zeros((3, m.nv)); mujoco.mj_jac(m, d, jac, None, cop, B('LF_palm_link'))
    tau = np.abs((d.qfrc_bias - jac.T @ np.array([0, 0, mass * g]))[dadr])
    Rp = d.xmat[PALM].reshape(3, 3)
    edge_down = float(-(Rp @ UP_LOCAL)[2])                       # 1 = upper edge facing the floor
    axis = Rp @ AXIS_LOCAL
    axis_tilt = abs(float(axis[2]))                              # 0 = claw lying horizontal
    along = float(np.ptp(patch[:, :2] @ (axis[:2] / (np.linalg.norm(axis[:2]) + 1e-9)))) if len(patch) else 0.0
    return dict(margin=margin, length=along, clearance=clearance, com_z=float(com[2]), tau=tau,
                tip_only=tip_only, n_patch=len(patch), edge_down=edge_down, axis_tilt=axis_tilt)


def score(x):
    r = evaluate(x)
    return (30 * min(r['margin'], 0.012) - 80 * max(0.0, -r['margin'])
            + 20 * min(r['length'], 0.05) - 200 * max(0.0, 0.03 - r['clearance'])
            + 6 * min(r['com_z'], 0.20) - 3 * r['tip_only']
            + 4 * min(r['edge_down'], 0.8) - 6 * max(0.0, r['axis_tilt'] - 0.25)
            - r['tau'].max() / PLATEAU_TORQUE - 0.2 * r['tau'].mean() / PLATEAU_TORQUE
            - 30 * max(0.0, r['tau'].max() / PLATEAU_TORQUE - 0.65))     # held: stay under the 1.2 N*m continuous rating


def search(gens=350, pop=64, seed=0):
    rs_ = np.random.default_rng(seed)
    lo = np.r_[-np.pi, -np.pi, rng[:, 0]]; hi = np.r_[np.pi, np.pi, rng[:, 1]]
    mean = np.r_[rs_.uniform(-1.5, 1.5), rs_.uniform(-1.5, 0.5), [fp.HOME[n] for n in NAMES]]
    sigma = (hi - lo) * 0.25; best = (-np.inf, None)
    for gen in range(gens):
        X = np.clip(mean + sigma * rs_.standard_normal((pop, len(lo))), lo, hi); X[0] = mean
        S = np.array([score(x) for x in X]); o = np.argsort(-S)
        w = np.log(pop / 4 + 0.5) - np.log(np.arange(1, pop // 4 + 1)); w /= w.sum()
        el = X[o[:pop // 4]]; mean = w @ el
        sigma = np.clip(np.sqrt(w @ (el - mean) ** 2), (hi - lo) * 0.002, (hi - lo) * 0.3)
        if S[o[0]] > best[0]:
            best = (S[o[0]], X[o[0]].copy())
    return best


if __name__ == '__main__':
    from multiprocessing import Pool
    with Pool(8) as pool:
        res = pool.starmap(search, [(500, 96, s) for s in range(8)])
    out = []
    for seed, (s, x) in enumerate(res):
        r = evaluate(x); i = int(np.argmax(r['tau']))
        out.append({'seed': seed, 'score': float(s), 'x': x.tolist()})
        print(f"seed {seed}: edge down {r['edge_down']:+.2f}, claw tilt {r['axis_tilt']:.2f}, support {r['length']*100:.1f} cm along the claw ({r['n_patch']} pts{', TIP ONLY' if r['tip_only'] else ''}), "
              f"CoM {r['margin']*1000:+.1f} mm inside, body CoM {r['com_z']*100:.0f} cm high, others clear {r['clearance']*1000:.0f} mm, "
              f"worst {NAMES[i]} {100*r['tau'][i]/PLATEAU_TORQUE:.0f}%", flush=True)
    Path('freeze_edge.json').write_text(json.dumps(out, indent=1))
