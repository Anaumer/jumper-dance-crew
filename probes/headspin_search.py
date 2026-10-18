"""Open-loop search for a backspin ("headspin" on the shell): lie on the back, push round with
the claws for three turns, flip back onto the feet. Same physics as move_search.py (Train asset,
kp=10/kd=0.5 through the measured servo curve); every frame is what MuJoCo produced.

Friction: the floor and every limb keep the asset's mu=1. The shell's collision geom gets
SHELL_MU (env, default 0.35) -- a printed PLA/PETG shell on a dance floor, not rubber. The
asset's collision class has priority=1, so the shell's own value is the one MuJoCo uses.

stages (each starts from a recorded state, so they chain):
  toback  from standing at HOME: end lying on the back, still
  spin    from the back: K_CYC keyframes repeated -- yaw about the world z, three turns
  flipup  from the end of the spin: end on the feet, upright, still

usage: headspin_search.py <stage> <generations> <seed> [init.json]  -> hs_<stage>_best_<seed>.json
       headspin_search.py replay <best.json> <out.npz>            (also writes the end state)
"""
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np

import roll_search as rs
import move_search as ms
from roll_search import HOME, STAND_Z, NAMES

SHELL_MU = float(os.environ.get('SHELL_MU', '0.35'))
HERE = Path(__file__).parent
TURNS = 3
K_CYC = 4                      # keyframes in one spin cycle
T = {'toback': 3.0, 'spin': 7.0, 'flipup': 3.5}
START = {'toback': None, 'spin': os.environ.get('SPIN_START', 'hs_back_state.npz'),
         'flipup': os.environ.get('FLIP_START', 'hs_spun_state.npz')}

_m = None


def model():
    global _m
    if _m is None:
        m, qadr, dadr, rng = rs.model()
        for name in ('upper_shell_link_meshcol', 'base_link_meshcol', 'display_module_link_meshcol'):
            g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
            if g >= 0:
                m.geom_friction[g, 0] = SHELL_MU
        _m = (m, qadr, dadr, rng)
    return _m


def n_keys(stage):
    return K_CYC if stage == 'spin' else ms.K


def bounds(stage):
    k = n_keys(stage)
    lo = np.tile([0.06] + [ms.B[s][0] for s in ms.SIDE_KEYS] * 2, k)
    hi = np.tile([0.70 if stage != 'spin' else 0.45] + [ms.B[s][1] for s in ms.SIDE_KEYS] * 2, k)
    return lo, hi


def keyframes(x, stage):
    """[(duration, targets)] for the stage's keyframes."""
    out = []
    for k in range(n_keys(stage)):
        v = x[k * ms.DIM_KF:(k + 1) * ms.DIM_KF]
        q = dict(HOME); n = len(ms.SIDE_KEYS)
        q.update(ms.side_targets('L', dict(zip(ms.SIDE_KEYS, v[1:1 + n]))))
        q.update(ms.side_targets('R', dict(zip(ms.SIDE_KEYS, v[1 + n:1 + 2 * n]))))
        out.append((float(v[0]), q))
    return out


def target_at(kf, stage, t, t_spin_end=None):
    """Spin: cycle the keyframes until t_spin_end, then HOME (arms tucked). Else: sequential."""
    if stage == 'spin':
        if t_spin_end is not None and t >= t_spin_end:
            return dict(HOME)
        period = sum(d for d, _ in kf)
        tau = t % period
        for d, q in kf:
            if tau < d:
                return q
            tau -= d
        return kf[-1][1]
    tau = t
    for d, q in kf:
        if tau < d:
            return q
        tau -= d
    return dict(HOME)


def initial(stage):
    m, qadr, dadr, rng = model()
    d = mujoco.MjData(m)
    if START[stage] is None:
        d.qpos[2] = STAND_Z; d.qpos[3] = 1
        d.qpos[qadr] = [HOME[n] for n in NAMES]
    else:
        s = np.load(HERE / START[stage])
        d.qpos[:] = s['qpos']; d.qvel[:] = s['qvel']
    mujoco.mj_forward(m, d)
    return m, d


def yaw_of(R):
    """Heading of the body's long axis in the world xy-plane (well defined on the back too)."""
    return np.arctan2(R[1, 0], R[0, 0])


def rollout(x, stage, record=False):
    m, qadr, dadr, rng = model()
    m, d = initial(stage)
    kf = keyframes(x, stage)
    x0 = d.qpos[:2].copy()
    total = 0.0; prev = yaw_of(d.xmat[1].reshape(3, 3)); rec = []
    t_done = None; on_back = 0.0; nb = 0; tilt = 0.0; peak_up = -1.0
    t_spin_end = None
    for i in range(int(T[stage] / rs.DT)):
        t = i * rs.DT
        if i % 2 == 0:
            tq = target_at(kf, stage, t, t_spin_end)
            target = np.clip([tq[k] for k in NAMES], rng[:, 0], rng[:, 1])
        qd = d.qvel[dadr]
        tau = rs.KP * (target - d.qpos[qadr]) - rs.KD * qd
        lim = rs.torque_limit(qd)
        d.qfrc_applied[:] = 0
        d.qfrc_applied[dadr] = np.clip(tau, -lim, lim)
        mujoco.mj_step(m, d)
        if i % 10 == 0:
            R = d.xmat[1].reshape(3, 3)
            y = yaw_of(R); peak_up = max(peak_up, R[2, 2])
            total += (y - prev + np.pi) % (2 * np.pi) - np.pi; prev = y
            if stage == 'spin':
                on_back += float(R[2, 2] < -0.8); nb += 1
                tilt += 1 + R[2, 2]
                if t_done is None and abs(total) >= TURNS * 2 * np.pi:
                    t_done = t; t_spin_end = t     # three turns done: stop pushing, settle
            if record:
                rec.append(d.qpos.copy())
        if not np.isfinite(d.qpos).all():
            return -1e3, {}, None, None
    R = d.xmat[1].reshape(3, 3)
    up = R[2, 2]; still = float(np.linalg.norm(d.qvel[:6]))
    drift = float(np.linalg.norm(d.qpos[:2] - x0))
    end = (d.qpos.copy(), d.qvel.copy())
    if stage == 'toback':
        score = 6 * (-up) - 1.0 * still - 2 * drift
        info = dict(up=round(float(up), 3), still=round(still, 3), drift=round(drift, 3))
    elif stage == 'spin':
        goal = TURNS * 2 * np.pi
        frac = min(abs(total), goal) / goal
        score = 20 * frac + (6 * (T[stage] - t_done) / T[stage] if t_done else 0) \
            + 4 * on_back / max(nb, 1) - 3 * tilt / max(nb, 1) - 4 * drift \
            + (2 * (-up) - 0.5 * still if t_done else 0)
        info = dict(turns=round(float(total / (2 * np.pi)), 2), t_done=t_done and round(t_done, 2),
                    on_back=round(on_back / max(nb, 1), 3), drift=round(drift, 3), up=round(float(up), 3))
    else:
        landed = (up > 0.9) * max(0.0, 1 - abs(d.qpos[2] - STAND_Z) / 0.03)
        # peak_up shapes the search: lying still on the back scores the same everywhere else
        score = 8 * up + 6 * landed - 1.0 * still - 1 * drift + 4 * peak_up
        info = dict(up=round(float(up), 3), peak_up=round(float(peak_up), 3), height=round(float(d.qpos[2]), 3),
                    still=round(still, 3), drift=round(drift, 3))
    return float(score), info, (np.array(rec) if record else None), end


def _eval(args):
    x, stage = args
    s, info, _, _ = rollout(x, stage)
    return s, info


def home_x(stage):
    return np.tile([0.3] + [ms.HOME_L[k] for k in ms.SIDE_KEYS] * 2, n_keys(stage))


def search(stage, gens, seed, init=None, pop=96, workers=int(os.environ.get('WORKERS', '14'))):
    rng = np.random.default_rng(seed)
    LO, HI = bounds(stage)
    mean = np.clip(init if init is not None else home_x(stage), LO, HI)
    if init is None and stage != 'toback':
        mean = np.clip(mean + (HI - LO) * 0.1 * rng.standard_normal(len(LO)), LO, HI)
    sigma = (HI - LO) * (0.12 if init is not None else 0.25)
    best = (-np.inf, None, None)
    log = open(HERE / f'hs_{stage}_log_{seed}.txt', 'w')
    with Pool(workers) as pool:
        for g in range(gens):
            X = np.clip(mean + sigma * rng.standard_normal((pop, len(LO))), LO, HI)
            X[0] = mean
            res = pool.map(_eval, [(x, stage) for x in X])
            S = np.array([r[0] for r in res]); o = np.argsort(-S)
            w = np.log(pop / 4 + 0.5) - np.log(np.arange(1, pop // 4 + 1)); w /= w.sum()
            elite = X[o[:pop // 4]]
            mean = w @ elite
            sigma = np.clip(np.sqrt(w @ (elite - mean) ** 2), (HI - LO) * 0.01, (HI - LO) * 0.3)
            if S[o[0]] > best[0]:
                best = (S[o[0]], X[o[0]].copy(), res[o[0]][1])
                (HERE / f'hs_{stage}_best_{seed}.json').write_text(json.dumps(
                    {'stage': stage, 'shell_mu': SHELL_MU, 'score': best[0], 'info': best[2],
                     'x': best[1].tolist()}, indent=1))
            print(f'gen {g:3d} best {S[o[0]]:6.2f} {res[o[0]][1]} | overall {best[0]:6.2f} {best[2]}',
                  file=log, flush=True)
    return best


if __name__ == '__main__':
    if sys.argv[1] == 'replay':
        b = json.loads(Path(sys.argv[2]).read_text())
        s, info, rec, end = rollout(np.array(b['x']), b['stage'], record=True)
        np.savez(sys.argv[3], qpos=rec, dt=rs.DT * 10)
        np.savez(Path(sys.argv[3]).with_name(Path(sys.argv[3]).stem + '_end.npz'), qpos=end[0], qvel=end[1])
        print(json.dumps({'score': s, **info}))
    else:
        stage, gens, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
        init = np.array(json.loads(Path(sys.argv[4]).read_text())['x']) if len(sys.argv) > 4 else None
        s, x, info = search(stage, gens, seed, init)
        print(json.dumps({'stage': stage, 'score': s, 'info': info}))
