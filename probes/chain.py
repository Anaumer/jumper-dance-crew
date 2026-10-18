"""Chain recorded moves into one continuous physics simulation (one breakdance routine).

Each move is an open-loop target schedule found by roll_search.py / move_search.py, both of
which start from rest at HOME. Here they run back to back in ONE simulation: move k+1 starts at
an absolute time chosen by the caller (snapped to the beat), from whatever state move k really
left -- so every frame is physics, and a start that comes too early simply shows up as a move
that fails, which `check` reports per move.
"""
import json
from pathlib import Path

import mujoco
import numpy as np

import roll_search as rs
import move_search as ms
from roll_search import HOME, STAND_Z, NAMES

MOVES = Path(__file__).parent / 'moves'


ARM_SIGN = {0: -1, 1: 1, 2: -1, 3: -1, 4: -1}


def mirror(q):
    """Left-right mirror of a target dict: swap sides, with the sign rule HOME obeys
    (legs negate every joint; arms negate J0, J2, J3, J4 and keep J1)."""
    out = {}
    for n, v in q.items():
        other = ('R' if n[0] == 'L' else 'L') + n[1:]
        k = int(n[4])
        out[other] = (ARM_SIGN[k] if n[1] == 'F' else -1) * v
    return out


class Move:
    def __init__(self, name, mirrored=False):
        self.name, self.mirrored = name, mirrored
        b = json.loads((MOVES / f'{name}.json').read_text())
        if 'params' in b:                                   # roll_search format
            self.kind, self.p = 'phase', b['params']
        else:                                               # move_search format
            self.kind, self.sched = 'keys', ms.schedule(np.array(b['x']))

    def targets(self, tau):
        if self.kind == 'phase':
            q = rs.targets(self.p, tau) if tau >= 0 else dict(HOME)
        else:
            q = dict(HOME)
            for t0, tq in self.sched:
                if tau >= t0:
                    q = tq
        return mirror(q) if self.mirrored else q


def simulate(moves, starts, t_end, record_every=0.01):
    m, qadr, dadr, rng = rs.model()
    d = mujoco.MjData(m)
    d.qpos[2] = STAND_Z; d.qpos[3] = 1
    d.qpos[qadr] = [HOME[n] for n in NAMES]
    mujoco.mj_forward(m, d)
    rec, upright, speed = [], [], []
    every = int(round(record_every / rs.DT))
    for i in range(int(t_end / rs.DT)):
        t = i * rs.DT
        k = max([j for j, s in enumerate(starts) if t >= s], default=None)
        tq = moves[k].targets(t - starts[k]) if k is not None else dict(HOME)
        target = np.clip([tq[n] for n in NAMES], rng[:, 0], rng[:, 1])
        qd = d.qvel[dadr]
        tau = rs.KP * (target - d.qpos[qadr]) - rs.KD * qd
        lim = rs.torque_limit(qd)
        d.qfrc_applied[:] = 0
        d.qfrc_applied[dadr] = np.clip(tau, -lim, lim)
        mujoco.mj_step(m, d)
        if i % every == 0:
            R = d.xmat[1].reshape(3, 3)
            rec.append(d.qpos.copy()); upright.append(R[2, 2])
            speed.append(float(np.linalg.norm(d.qvel[:6])))
    return np.array(rec), np.array(upright), np.array(speed)


def settle_time(upright, speed, after, dt=0.01, need=0.15):
    """First time after `after` the robot has been upright and nearly still for `need` seconds."""
    ok = (upright > 0.97) & (speed < 0.15)
    run = 0
    for i in range(int(after / dt), len(ok)):
        run = run + 1 if ok[i] else 0
        if run * dt >= need:
            return (i + 1) * dt - need
    return None
