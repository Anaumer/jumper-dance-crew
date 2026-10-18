"""Probe: is a one-claw handstand freeze statically possible for Jumper?

Searches a pose (base roll/pitch + all 22 joints) in which ONLY the left claw touches the floor,
the CoM sits exactly over that contact, and the joint torques gravity demands (with the whole
weight on the claw) are as small as possible. Static torques: tau = g(q)_joints - (J_c^T F)_joints
with F = (0, 0, m g) at the contact point -- equilibrium of every joint given the contact force.
"""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

import roll_search as rs
from roll_search import HOME, NAMES
from tasks.jumper.common.actuator import PLATEAU_TORQUE

m, qadr, dadr, rng = rs.model()
d = mujoco.MjData(m)
B = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
mass = float(m.body_mass.sum()); g = 9.81
COL = [gi for gi in range(m.ngeom) if m.geom_contype[gi] and m.geom_bodyid[gi] > 0 and m.geom_type[gi] == 7]
VERT = {gi: m.mesh_vert[m.mesh_vertadr[m.geom_dataid[gi]]:m.mesh_vertadr[m.geom_dataid[gi]] + m.mesh_vertnum[m.geom_dataid[gi]]] for gi in COL}


def in_subtree(b, root):
    while b > 0:
        if b == root:
            return True
        b = m.body_parentid[b]
    return False


SUPPORT = [gi for gi in COL if in_subtree(m.geom_bodyid[gi], B('LF_palm_link'))]   # the claw (palm + finger)
OTHER = [gi for gi in COL if gi not in SUPPORT]


def pose(x):
    d.qpos[:] = 0; d.qpos[2] = 1.0
    d.qpos[3:7] = rs.np.array([1, 0, 0, 0]) if False else 0
    q = np.zeros(4); mujoco.mju_euler2Quat(q, np.array([x[0], x[1], 0.0]), 'xyz'); d.qpos[3:7] = q
    d.qpos[qadr] = np.clip(x[2:], rng[:, 0], rng[:, 1])
    mujoco.mj_forward(m, d)
    lows = {gi: (VERT[gi] @ d.geom_xmat[gi].reshape(3, 3).T + d.geom_xpos[gi]) for gi in COL}
    sup = np.concatenate([lows[gi] for gi in SUPPORT]); c = sup[np.argmin(sup[:, 2])]
    d.qpos[2] -= c[2]; mujoco.mj_forward(m, d)                 # contact point on the floor
    clearance = min(float((VERT[gi] @ d.geom_xmat[gi].reshape(3, 3).T + d.geom_xpos[gi])[:, 2].min()) for gi in OTHER)
    c = c.copy(); c[2] = 0.0
    com = d.subtree_com[1]
    jac = np.zeros((3, m.nv)); mujoco.mj_jac(m, d, jac, None, c, B('LF_palm_link'))
    F = np.array([0, 0, mass * g])
    tau = d.qfrc_bias - jac.T @ F                              # residual the joints must supply
    return dict(com_off=float(np.linalg.norm(com[:2] - c[:2])), clearance=clearance,
                tau=np.abs(tau[dadr]), base_resid=float(np.abs(tau[:6]).max()))


def score(x):
    r = pose(x)
    return (-20 * r['com_off'] - 200 * max(0.0, 0.03 - r['clearance'])
            - r['tau'].max() / PLATEAU_TORQUE - 0.2 * r['tau'].mean() / PLATEAU_TORQUE)


def search(gens=300, pop=64, seed=0):
    rs_ = np.random.default_rng(seed)
    lo = np.r_[-np.pi, -np.pi, rng[:, 0]]; hi = np.r_[np.pi, np.pi, rng[:, 1]]
    mean = np.r_[0.0, -1.2, [HOME[n] for n in NAMES]]; sigma = (hi - lo) * 0.2
    best = (-np.inf, None)
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
    results = []
    for seed in range(4):
        s, x = search(seed=seed)
        r = pose(x)
        i = int(np.argmax(r['tau']))
        results.append((s, x))
        print(f'seed {seed}: CoM off contact {r["com_off"]*1000:.1f} mm, other limbs clear {r["clearance"]*1000:.0f} mm, '
              f'worst joint {NAMES[i]} {r["tau"][i]:.2f} N*m ({100*r["tau"][i]/PLATEAU_TORQUE:.0f}% of plateau), '
              f'base residual {r["base_resid"]:.3f}', flush=True)
    s, x = max(results, key=lambda r: r[0])
    Path('freeze_pose.json').write_text(json.dumps({'x': x.tolist(), 'names': NAMES}, indent=1))
