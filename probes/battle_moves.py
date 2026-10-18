"""Battle gestures as open-loop joint-target schedules (same PD + servo curve as every move).

Every segment starts and ends at rest at HOME, like the acrobatic moves, so chain.simulate can put
them back to back. Arm poses are solved by IK at the HOME stance: a claw tip (the arm's foot site)
placed at a point in the body frame.
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

import roll_search as rs
from roll_search import HOME, NAMES, STAND_Z

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'choreo'))

BPM = 108
BEAT = 60 / BPM
ARM_SIGN = {0: -1, 1: 1, 2: -1, 3: -1, 4: -1}


def smooth(u):
    u = np.clip(u, 0, 1)
    return u * u * u * (10 - 15 * u + 6 * u * u)


def arm_ik(side, target_body, j4=0.0):
    """Arm joints (J0..J3) putting that arm's claw site at `target_body` (body frame, HOME stance)."""
    m = mujoco.MjModel.from_xml_path(str(rs.ROBOT / 'jumper/assets/jumper/jumper.xml'))
    d = mujoco.MjData(m)
    d.qpos[2] = STAND_Z; d.qpos[3] = 1
    J = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    for n, v in HOME.items():
        d.qpos[m.jnt_qposadr[J(n)]] = v
    mujoco.mj_forward(m, d)
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f'{side}F')
    names = [f'{side}F_J{k}_joint' for k in range(4)]
    adr = [m.jnt_qposadr[J(n)] for n in names]; dof = [m.jnt_dofadr[J(n)] for n in names]
    goal = np.array([0, 0, STAND_Z]) + np.asarray(target_body)
    # multi-start: refine the 30 closest of 3000 random arm configurations and keep the best.
    # From HOME alone the damped least squares stalls (measured: 19-32 cm short of reachable
    # targets); one seed alone still runs into joint stops near the top of the workspace.
    rng_ = np.random.default_rng(0); samples = []
    for _ in range(3000):
        q = [rng_.uniform(*m.jnt_range[J(n)]) for n in names]
        for a_, v in zip(adr, q):
            d.qpos[a_] = v
        mujoco.mj_kinematics(m, d)
        samples.append((np.linalg.norm(goal - d.site_xpos[site]), q))
    samples.sort(key=lambda r: r[0])
    jac = np.zeros((3, m.nv)); best = (np.inf, None)
    for _, q in samples[:30]:
        for a_, v in zip(adr, q):
            d.qpos[a_] = v
        mujoco.mj_kinematics(m, d)
        for _ in range(150):
            err = goal - d.site_xpos[site]
            if np.linalg.norm(err) < 1e-5:
                break
            mujoco.mj_jacSite(m, d, jac, None, site)
            Jl = jac[:, dof]
            dq = Jl.T @ np.linalg.solve(Jl @ Jl.T + 1e-4 * np.eye(3), err)
            for a_, n, v in zip(adr, names, dq):
                d.qpos[a_] = np.clip(d.qpos[a_] + 0.5 * v, *m.jnt_range[J(n)])
            mujoco.mj_kinematics(m, d)
        e = np.linalg.norm(goal - d.site_xpos[site])
        if e < best[0]:
            best = (e, [float(d.qpos[a_]) for a_ in adr])
    for a_, v in zip(adr, best[1]):
        d.qpos[a_] = v
    mujoco.mj_kinematics(m, d)
    q = {n: float(d.qpos[a]) for n, a in zip(names, adr)}
    q[f'{side}F_J4_joint'] = (1 if side == 'L' else -1) * j4
    return q, float(np.linalg.norm(goal - d.site_xpos[site]))


class Keyframes:
    """Piecewise quintic-smooth joint targets between (time, partial joint dict) keys."""

    def __init__(self, keys, name='keys'):
        self.name = name
        self.keys = []
        q = dict(HOME)
        for t, upd in keys:
            q = dict(q); q.update(upd); self.keys.append((t, q))
        self.duration = keys[-1][0]

    def targets(self, tau):
        if tau <= 0 or tau >= self.duration:
            return dict(HOME) if tau <= 0 else dict(self.keys[-1][1])
        for (t0, q0), (t1, q1) in zip(self.keys, self.keys[1:]):
            if t0 <= tau < t1:
                u = smooth((tau - t0) / (t1 - t0))
                return {n: q0[n] + (q1[n] - q0[n]) * u for n in HOME}
        return dict(self.keys[-1][1])


def bob_offsets(tau, depth=0.12):
    """Knee-and-hip bounce on every beat: the body dips on the beat and rises between."""
    s = depth * (0.5 - 0.5 * np.cos(2 * np.pi * tau / BEAT))
    out = {}
    for leg in ('LM', 'LR'):
        out[f'{leg}_J1_joint'] = HOME[f'{leg}_J1_joint'] + s
    for leg in ('RM', 'RR'):
        out[f'{leg}_J1_joint'] = HOME[f'{leg}_J1_joint'] - s
    return out


class Bob:
    """Stand and bob to the beat for n beats, arms in a pose (or HOME)."""

    def __init__(self, beats, arms=None, extra=None, name='bob'):
        self.name, self.duration, self.arms = name, beats * BEAT, arms or {}
        self.extra = extra

    def targets(self, tau):
        q = dict(HOME)
        if 0 <= tau < self.duration:
            ramp = smooth(min(tau, self.duration - tau) / (0.5 * BEAT))
            for n, v in bob_offsets(tau).items():
                q[n] = HOME[n] + (v - HOME[n]) * ramp
            for n, v in self.arms.items():
                q[n] = HOME[n] + (v - HOME[n]) * ramp
            if self.extra:
                for n, v in self.extra(tau).items():
                    q[n] = HOME[n] + (v - HOME[n]) * ramp
        return q


# arm poses (body frame targets for the claw sites)
def poses():
    up_l, e1 = arm_ik('L', [0.07, 0.03, 0.135])      # claws up over the head (the arm's reach: ~0.14 m above the body)
    up_r, e2 = arm_ik('R', [0.07, -0.03, 0.135])
    eyes_l, e3 = arm_ik('L', [0.16, 0.015, 0.06])     # claws in front of the face (display)
    eyes_r, e4 = arm_ik('R', [0.16, -0.015, 0.06])
    jab_l, e5 = arm_ik('L', [0.25, 0.03, 0.03], j4=-1.2)   # left arm thrust forward, claw open
    wide_l, e6 = arm_ik('L', [0.06, 0.11, 0.12])      # for the clap: apart
    wide_r, e7 = arm_ik('R', [0.06, -0.11, 0.12])
    return dict(up=(up_l | up_r), eyes=(eyes_l | eyes_r), jab=jab_l, wide=(wide_l | wide_r)), \
        dict(up=max(e1, e2), eyes=max(e3, e4), jab=e5, wide=max(e6, e7))


P, P_ERR = poses()


def clap(beats):
    """Bob with claws overhead, clapping together on every beat."""
    def extra(tau):
        u = 0.5 - 0.5 * np.cos(2 * np.pi * tau / BEAT)          # 0 together on the beat, 1 apart
        return {n: P['up'][n] + (P['wide'][n] - P['up'][n]) * u for n in P['up']}
    return Bob(beats, extra=extra, name='clap')


def cover_eyes(beats):
    return Bob(beats, arms=P['eyes'], name='cover_eyes')


def pinch(beats):
    """Left claw thrust toward the opponent, snapping on every half beat."""
    def extra(tau):
        q = dict(P['jab'])
        q['LF_J4_joint'] = -1.2 * (0.5 + 0.5 * np.cos(2 * np.pi * tau / (BEAT / 2)))
        return q
    return Bob(beats, extra=extra, name='pinch')
