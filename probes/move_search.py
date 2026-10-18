"""Search open-loop keyframe schedules for acrobatic moves in physics (Jumper, real servo curve).

A schedule is K keyframes; each keyframe holds a duration and, per side (L, R), joint targets
for the middle leg (3), the rear leg (3) and the arm (J0..J3), written in the LEFT side's
convention and mirrored for the right. Targets are held from one keyframe to the next (a PD
servo step, like a dancer's beat); after the last keyframe the robot is commanded to HOME.

modes
  barrel   360 deg about the long (x) axis, landing on the feet facing the same way
  forward  360 deg about the lateral (y) axis (a forward roll), same landing
Score: completed rotation (net, unwrapped) dominates, the furthest rotation shapes the search
before that, then upright / on its feet / still / heading at the end; rotation about the other
horizontal axis (tumbling) is charged.

usage: move_search.py <mode> <generations> <seed> [init.json]   -> <mode>_best_<seed>.json
       move_search.py replay <best.json> <states.npz>
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np

import roll_search as rs                       # model, servo curve, HOME
from roll_search import HOME, STAND_Z, NAMES

K = 8
SIDE_KEYS = ['m0', 'm1', 'm2', 'r0', 'r1', 'r2', 'a0', 'a1', 'a2', 'a3']
# per-joint bounds in the left convention (from the joint ranges, slightly inside)
B = {'m0': (-0.7, 0.95), 'm1': (-2.35, 1.45), 'm2': (-2.35, 1.75),
     'r0': (-0.55, 2.7), 'r1': (-2.35, 1.55), 'r2': (-2.35, 1.75),
     'a0': (-2.7, 0.65), 'a1': (-4.7, 1.35), 'a2': (-2.15, 1.6), 'a3': (-2.05, 1.6)}
DIM_KF = 1 + 2 * len(SIDE_KEYS)
LO = np.tile([0.06] + [B[k][0] for k in SIDE_KEYS] * 2, K)
HI = np.tile([0.70] + [B[k][1] for k in SIDE_KEYS] * 2, K)
T_START, T_END = 0.3, 5.0
import os
DRIFT_W = float(os.environ.get('DRIFT_W', '0'))   # charge for the body axis wandering off the move's axis (env, so pool workers see it)
HOME_L = {'m0': HOME['LM_J0_joint'], 'm1': HOME['LM_J1_joint'], 'm2': HOME['LM_J2_joint'],
          'r0': HOME['LR_J0_joint'], 'r1': HOME['LR_J1_joint'], 'r2': HOME['LR_J2_joint'],
          'a0': HOME['LF_J0_joint'], 'a1': HOME['LF_J1_joint'], 'a2': HOME['LF_J2_joint'],
          'a3': HOME['LF_J3_joint']}
ARM_SIGN = {0: -1, 1: 1, 2: -1, 3: -1}


def home_vector():
    kf = [0.3] + [HOME_L[k] for k in SIDE_KEYS] * 2
    return np.tile(kf, K)


def side_targets(side, vals):
    s = 1 if side == 'L' else -1
    q = {}
    for j in range(3):
        q[f'{side}M_J{j}_joint'] = s * vals[f'm{j}']
        q[f'{side}R_J{j}_joint'] = s * vals[f'r{j}']
    for j in range(4):
        q[f'{side}F_J{j}_joint'] = (1 if side == 'L' else ARM_SIGN[j]) * vals[f'a{j}']
    return q


def schedule(x):
    """[(t_start, targets dict)] from a parameter vector."""
    out, t = [], T_START
    for k in range(K):
        v = x[k * DIM_KF:(k + 1) * DIM_KF]
        q = dict(HOME)
        n = len(SIDE_KEYS)
        q.update(side_targets('L', dict(zip(SIDE_KEYS, v[1:1 + n]))))
        q.update(side_targets('R', dict(zip(SIDE_KEYS, v[1 + n:1 + 2 * n]))))
        out.append((t, q)); t += v[0]
    out.append((t, dict(HOME)))
    return out


_FREEZE = None


def freeze_target():
    global _FREEZE
    if _FREEZE is None:
        import freeze_pose as fp
        import freeze_flat as ff
        ff.place(np.array(json.loads(Path('freeze_target.json').read_text())['x']))   # open claw, flat-ish
        R = fp.d.xmat[1].reshape(3, 3).copy()
        claw = [gi for gi in fp.SUPPORT]
        _FREEZE = (R, set(claw))
    return _FREEZE


def freeze_quality(m, d):
    R_t, claw = freeze_target()
    R = d.xmat[1].reshape(3, 3)
    ground = [c for c in d.contact[:d.ncon] if 0 in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2])]
    if not ground:
        return 0.0
    only_claw = all((c.geom1 in claw) or (c.geom2 in claw) for c in ground)
    if not only_claw:
        return 0.0
    err = np.arccos(np.clip((np.trace(R_t.T @ R) - 1) / 2, -1, 1))       # rotation angle to the freeze
    return float(np.exp(-2.0 * err) * np.exp(-0.5 * np.linalg.norm(d.qvel[3:6])))


def angle_of(R, mode):
    zb = R[:, 2]
    if mode == 'barrel':
        return np.arctan2(-zb[1], zb[2]), abs(zb[0])       # rotation about x; tumble = z_b.x
    return np.arctan2(zb[0], zb[2]), abs(zb[1])            # rotation about y; tumble = z_b.y


def rollout(x, mode, record=False):
    m, qadr, dadr, rng = rs.model()
    d = mujoco.MjData(m)
    d.qpos[2] = STAND_Z; d.qpos[3] = 1
    d.qpos[qadr] = [HOME[n] for n in NAMES]
    mujoco.mj_forward(m, d)
    sched = schedule(x); si = -1
    target = np.array([HOME[n] for n in NAMES])
    total = 0.0; prev = 0.0; peak = 0.0; tumble = 0.0; nt = 0; rec = []; drift = 0.0; fq = 0.0; fq_peak = 0.0
    for i in range(int(T_END / rs.DT)):
        t = i * rs.DT
        while si + 1 < len(sched) and t >= sched[si + 1][0]:
            si += 1
            target = np.clip([sched[si][1][k] for k in NAMES], rng[:, 0], rng[:, 1])
        qd = d.qvel[dadr]
        tau = rs.KP * (target - d.qpos[qadr]) - rs.KD * qd
        lim = rs.torque_limit(qd)
        d.qfrc_applied[:] = 0
        d.qfrc_applied[dadr] = np.clip(tau, -lim, lim)
        mujoco.mj_step(m, d)
        if mode == 'freeze':
            if i % 10 == 0:
                fq += freeze_quality(m, d) * 0.01; fq_peak = max(fq_peak, freeze_quality(m, d))
                if record:
                    rec.append(d.qpos.copy())
            if not np.isfinite(d.qpos).all():
                return -1e3, {}, None
            continue
        if i % 10 == 0:
            R = d.xmat[1].reshape(3, 3)
            a, tb = angle_of(R, mode)
            total += (a - prev + np.pi) % (2 * np.pi) - np.pi
            prev = a; peak = max(peak, abs(total)); tumble += tb; nt += 1
            drift += 1.0 - (R[0, 0] if mode == 'barrel' else R[1, 1])   # turning off the move's axis
            if record:
                rec.append(d.qpos.copy())
        if not np.isfinite(d.qpos).all():
            return -1e3, {}, None
    R = d.xmat[1].reshape(3, 3)
    up = R[2, 2]; heading = R[0, 0]
    if mode == 'freeze':       # seconds spent in a good freeze dominate; landing on the feet is a bonus
        landed = (up > 0.9) * max(0.0, 1 - abs(d.qpos[2] - STAND_Z) / 0.03)    # back on all fours
        score = 20 * fq + 4 * fq_peak + 4 * up + 4 * landed - 0.5 * float(np.linalg.norm(d.qvel[:6]))
        return float(score), dict(freeze_s=round(fq, 3), freeze_peak=round(fq_peak, 3), upright=round(float(up), 3)), \
            (np.array(rec) if record else None)
    still = float(np.linalg.norm(d.qvel[:6]))
    full = 2 * np.pi
    score = 10 * min(abs(total), full) / full + 4 * min(peak, full) / full \
        + 3 * up + 2 * heading - 0.5 * still - 20 * abs(d.qpos[2] - STAND_Z) * (up > 0.8) \
        - 3 * max(0.0, abs(total) - full - 0.6) - 3 * tumble / max(nt, 1) \
        - DRIFT_W * drift / max(nt, 1)
    info = dict(rot_deg=round(float(np.degrees(total)), 1), peak_deg=round(float(np.degrees(peak)), 1),
                tumble=round(tumble / max(nt, 1), 3), drift=round(drift / max(nt, 1), 3), upright=round(float(up), 3),
                heading=round(float(heading), 3), height=round(float(d.qpos[2]), 3), still=round(still, 3))
    return float(score), info, (np.array(rec) if record else None)


def _eval(args):
    x, mode = args
    s, info, _ = rollout(x, mode)
    return s, info


def search(mode, gens, seed, init=None, pop=96, workers=6):
    rng = np.random.default_rng(seed)
    mean = np.clip(init if init is not None else home_vector(), LO, HI)
    sigma = (HI - LO) * (0.12 if init is not None else 0.25)
    best = (-np.inf, None, None)
    log = open(f'{mode}_log_{seed}.txt', 'w')
    with Pool(workers) as pool:
        for g in range(gens):
            X = np.clip(mean + sigma * rng.standard_normal((pop, len(LO))), LO, HI)
            X[0] = mean
            res = pool.map(_eval, [(x, mode) for x in X])
            S = np.array([r[0] for r in res]); o = np.argsort(-S)
            w = np.log(pop / 4 + 0.5) - np.log(np.arange(1, pop // 4 + 1)); w /= w.sum()
            elite = X[o[:pop // 4]]
            mean = w @ elite
            sigma = np.clip(np.sqrt(w @ (elite - mean) ** 2), (HI - LO) * 0.01, (HI - LO) * 0.3)
            if S[o[0]] > best[0]:
                best = (S[o[0]], X[o[0]].copy(), res[o[0]][1])
                Path(f'{mode}_best_{seed}.json').write_text(json.dumps(
                    {'mode': mode, 'score': best[0], 'info': best[2], 'x': best[1].tolist()}, indent=1))
            print(f'gen {g:3d} best {S[o[0]]:6.2f} {res[o[0]][1]} | overall {best[0]:6.2f} {best[2]}', file=log, flush=True)
    return best


if __name__ == '__main__':
    if sys.argv[1] == 'replay':
        b = json.loads(Path(sys.argv[2]).read_text())
        s, info, rec = rollout(np.array(b['x']), b['mode'], record=True)
        np.savez(sys.argv[3], qpos=rec, dt=rs.DT * 10)
        print(json.dumps({'score': s, **info}))
    else:
        mode, gens, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
        init = np.array(json.loads(Path(sys.argv[4]).read_text())['x']) if len(sys.argv) > 4 else None
        s, x, info = search(mode, gens, seed, init)
        print(json.dumps({'mode': mode, 'score': s, 'info': info}))
