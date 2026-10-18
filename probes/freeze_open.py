"""Probe: one-arm freeze on the OPENED claw lying flat on the floor (palm and finger both down).

Support area = convex hull of every claw vertex within 2 mm of the floor; the larger, the better.
CoM inside with margin (statically stable); the whole weight on the claw (centre of pressure under
the CoM) and every arm joint under 65% of the plateau, i.e. inside the 1.2 N*m continuous rating,
so the pose can be HELD; everything else >= 3 cm off the floor; the body held high.
"""
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

import freeze_pose as fp
import freeze_flat as ff
from freeze_pose import m, d, B, VERT, SUPPORT, OTHER, NAMES, qadr, rng
from tasks.jumper.common.actuator import PLATEAU_TORQUE

FINGER = B('LF_finger_link')
J4 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'LF_J4_joint')


def evaluate(x):
    ff.place(x)
    world = {gi: VERT[gi] @ d.geom_xmat[gi].reshape(3, 3).T + d.geom_xpos[gi] for gi in fp.COL}
    patch_by = {gi: world[gi][world[gi][:, 2] < 0.002] for gi in SUPPORT}
    patch = np.concatenate(list(patch_by.values()))
    finger_down = any(len(p) and fp.in_subtree(m.geom_bodyid[gi], FINGER) for gi, p in patch_by.items())
    palm_down = any(len(p) and not fp.in_subtree(m.geom_bodyid[gi], FINGER) for gi, p in patch_by.items())
    clear = min(float(world[gi][:, 2].min()) for gi in OTHER)
    com = d.subtree_com[1]
    area, margin = 0.0, -0.05
    if len(patch) >= 3:
        try:
            h = ConvexHull(patch[:, :2]); area = float(h.volume)
            margin = float(-(h.equations[:, :2] @ com[:2] + h.equations[:, 2]).max())
        except Exception:
            pass
    cop = np.array([com[0], com[1], 0.0])
    jac = np.zeros((3, m.nv)); mujoco.mj_jac(m, d, jac, None, cop, B('LF_palm_link'))
    tau = np.abs((d.qfrc_bias - jac.T @ np.array([0, 0, fp.mass * fp.g]))[fp.dadr])
    # the claw's plane is the one its finger opens in; flat on the floor = the finger's hinge axis vertical
    hinge = d.xmat[m.jnt_bodyid[J4]].reshape(3, 3) @ m.jnt_axis[J4]
    flat = abs(float(hinge[2]))
    return dict(flat=flat, area_cm2=area * 1e4, margin=margin, clear=clear, com_z=float(com[2]), tau=tau,
                both=finger_down and palm_down, j4=float(d.qpos[qadr[NAMES.index('LF_J4_joint')]]))


def score(x):
    r = evaluate(x)
    return (30 * min(r['margin'], 0.015) - 80 * max(0.0, -r['margin'])
            + 0.25 * min(r['area_cm2'], 40) + 3 * r['both'] + 8 * min(r['flat'], 0.95)
            - 200 * max(0.0, 0.03 - r['clear']) + 6 * min(r['com_z'], 0.20)
            - r['tau'].max() / PLATEAU_TORQUE - 30 * max(0.0, r['tau'].max() / PLATEAU_TORQUE - 0.65))


def search(gens=500, pop=96, seed=0):
    rs_ = np.random.default_rng(seed)
    lo = np.r_[-np.pi, -np.pi, rng[:, 0]]; hi = np.r_[np.pi, np.pi, rng[:, 1]]
    mean = np.r_[rs_.uniform(-1.6, 1.6), rs_.uniform(-1.2, 1.2), [fp.HOME[n] for n in NAMES]]
    mean[2 + NAMES.index('LF_J4_joint')] = -1.2                  # start with the claw open
    sigma = (hi - lo) * 0.25; best = (-np.inf, None)
    for _ in range(gens):
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
        r = evaluate(x); i = int(np.argmax(r['tau'])); out.append({'seed': seed, 'score': float(s), 'x': x.tolist()})
        print(f"seed {seed}: flat {r['flat']:.2f}, area {r['area_cm2']:.1f} cm2 ({'palm+finger' if r['both'] else 'one part'}), claw J4 {r['j4']:+.2f}, "
              f"CoM {r['margin']*1000:+.1f} mm inside, body CoM {r['com_z']*100:.0f} cm, rest clear {r['clear']*1000:.0f} mm, "
              f"worst {NAMES[i]} {100*r['tau'][i]/PLATEAU_TORQUE:.0f}%", flush=True)
    Path('freeze_open_flat.json').write_text(json.dumps(out, indent=1))
