"""The whole headspin as ONE continuous simulation: lie down (toback schedule), spin (the spin
cycle, until three turns), let the spin run out with the arms tucked, flip up (flipup schedule).
The stage switches happen on the measured state, so every frame follows from the one before.

usage: headspin_chain.py <toback.json> <spin.json> <flipup.json> <out.npz>
"""
import json
import sys

import mujoco
import numpy as np

import headspin_search as hs
import roll_search as rs
from roll_search import HOME, NAMES

kf = {s: hs.keyframes(np.array(json.load(open(f))['x']), s)
      for s, f in zip(('toback', 'spin', 'flipup'), sys.argv[1:4])}
m, qadr, dadr, rng = hs.model()
d = mujoco.MjData(m)
d.qpos[2] = hs.STAND_Z; d.qpos[3] = 1
d.qpos[qadr] = [HOME[n] for n in NAMES]
mujoco.mj_forward(m, d)

stage, t_stage, total, prev = 'toback', 0.0, 0.0, None
log, rec, ups = [], [], []
target = np.array([HOME[n] for n in NAMES])
for i in range(int(12.0 / rs.DT)):
    t = i * rs.DT
    R = d.xmat[1].reshape(3, 3); up = R[2, 2]; w = float(np.linalg.norm(d.qvel[3:6]))
    y = hs.yaw_of(R)
    if prev is not None and stage in ('spin', 'runout'):
        total += (y - prev + np.pi) % (2 * np.pi) - np.pi
    prev = y
    if stage == 'toback' and t - t_stage > 0.8 and up < -0.95 and w < 0.4:
        stage, t_stage = 'spin', t; log.append(('spin', t))
    elif stage == 'spin' and abs(total) >= hs.TURNS * 2 * np.pi:
        stage, t_stage = 'runout', t; log.append(('runout', t))
    elif stage == 'runout' and abs(d.qvel[5]) < 0.6 and t - t_stage > 0.15:
        stage, t_stage = 'flipup', t; log.append(('flipup', t))
    elif stage == 'flipup' and t - t_stage > 1.0 and up > 0.97 and float(np.linalg.norm(d.qvel[:6])) < 0.15:
        log.append(('done', t)); break
    if i % 2 == 0:
        tau = t - t_stage
        if stage == 'toback':
            tq = hs.target_at(kf['toback'], 'toback', tau)
        elif stage == 'spin':
            tq = hs.target_at(kf['spin'], 'spin', tau)
        elif stage == 'runout':
            tq = dict(HOME)
        else:
            tq = hs.target_at(kf['flipup'], 'flipup', tau)
        target = np.clip([tq[k] for k in NAMES], rng[:, 0], rng[:, 1])
    qd = d.qvel[dadr]
    tau_ = rs.KP * (target - d.qpos[qadr]) - rs.KD * qd
    lim = rs.torque_limit(qd)
    d.qfrc_applied[:] = 0
    d.qfrc_applied[dadr] = np.clip(tau_, -lim, lim)
    mujoco.mj_step(m, d)
    if i % 10 == 0:
        rec.append(d.qpos.copy()); ups.append(up)
rec = np.array(rec)
np.savez(sys.argv[4], qpos=rec, dt=rs.DT * 10, stages=np.array([s for s, _ in log]),
         times=np.array([t for _, t in log]))
print('stages', [(s, round(t, 2)) for s, t in log], f'turns {total / (2 * np.pi):.2f}',
      f'end up {ups[-1]:.3f} height {rec[-1, 2]:.3f} drift {np.linalg.norm(rec[-1, :2] - rec[0, :2]):.3f} m',
      f'length {len(rec) * 0.01:.2f} s')
