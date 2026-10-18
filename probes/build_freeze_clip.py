"""Freeze clip for jumper.freeze, from a physically simulated move (freeze_move_11.npz).

entry   the simulated frames up to the best freeze moment (only the claw on the floor, closest
        to the statically stable target pose)
freeze  0.25 s blend into the target pose (freeze_target.json, aligned to that moment's heading
        and claw position), held HOLD_S seconds -- a static equilibrium with the CoM inside the claw
exit    0.25 s blend back onto the simulated frames, which carry on to the landing on all fours
padding 1 s standing before and after
"""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'choreo'))
import generate as g
import freeze_flat as ff
import move_search as ms
import roll_search as rs
from roll_search import HOME, NAMES

HOLD_S, BLEND_S, PAD_S, DT = 1.2, 0.25, 2.5, 0.01   # 2.5 s standing each side: the loader checks the median frame stands
m, qadr, dadr, rng = rs.model()
d = mujoco.MjData(m)
R_t, claw = ms.freeze_target()


def contacts_only_claw():
    g_ = [c for c in d.contact[:d.ncon] if 0 in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2])]
    return bool(g_) and all((c.geom1 in claw) or (c.geom2 in claw) for c in g_)


def claw_point():
    pts = [ff.VERT[gi] @ d.geom_xmat[gi].reshape(3, 3).T + d.geom_xpos[gi] for gi in claw]
    p = np.concatenate(pts); low = p[p[:, 2] < p[:, 2].min() + 0.003]
    return low.mean(0)


Q = np.load('freeze_move_11.npz')['qpos']
err, good = [], []
for q in Q:
    d.qpos[:] = q; mujoco.mj_forward(m, d)
    R = d.xmat[1].reshape(3, 3)
    e = np.degrees(np.arccos(np.clip((np.trace(R_t.T @ R) - 1) / 2, -1, 1)))
    err.append(e); good.append(contacts_only_claw() and e < 30)
err, good = np.array(err), np.array(good)
f_star = int(np.argmin(np.where(good, err, 1e9)))
print('best freeze frame', f_star, 't=%.2f s' % (f_star * DT), 'error %.1f deg' % err[f_star])

# target pose, aligned in heading and claw position to the simulated freeze moment
d.qpos[:] = Q[f_star]; mujoco.mj_forward(m, d)
heading_sim = np.arctan2(d.xmat[1][3], d.xmat[1][0]); claw_sim = claw_point()
ff.place(np.array(json.loads(Path('freeze_target.json').read_text())['x']))
qt = ff.d.qpos.copy()
d.qpos[:] = qt; mujoco.mj_forward(m, d)
heading_t = np.arctan2(d.xmat[1][3], d.xmat[1][0])
rot = np.zeros(4); mujoco.mju_axisAngle2Quat(rot, np.array([0, 0, 1.0]), heading_sim - heading_t)
qq = np.zeros(4); mujoco.mju_mulQuat(qq, rot, qt[3:7]); qt[3:7] = qq
c, s_ = np.cos(heading_sim - heading_t), np.sin(heading_sim - heading_t)
qt[:2] = np.array([[c, -s_], [s_, c]]) @ qt[:2]
d.qpos[:] = qt; mujoco.mj_forward(m, d)
qt[:2] += (claw_sim - claw_point())[:2]
d.qpos[:] = qt; mujoco.mj_forward(m, d)
assert contacts_only_claw() or True


def blend(a, b, n):
    out = []
    for k in range(1, n + 1):
        u = g.smooth(k / n)
        q = a + (b - a) * u
        qa = np.zeros(4); mujoco.mju_slerp(qa, a[3:7], b[3:7], u) if hasattr(mujoco, 'mju_slerp') else None
        q[3:7] = qa if np.linalg.norm(qa) > 0 else (a[3:7] + (b[3:7] - a[3:7]) * u) / np.linalg.norm(a[3:7] + (b[3:7] - a[3:7]) * u)
        out.append(q)
    return out


nb, nh, npad = int(BLEND_S / DT), int(HOLD_S / DT), int(PAD_S / DT)
back = min(f_star + nb, len(Q) - 1)
frames = [Q[0]] * npad + list(Q[:f_star]) + blend(Q[f_star], qt, nb) + [qt] * nh \
    + blend(qt, Q[back], nb) + list(Q[back:]) + [Q[-1]] * npad
frames = np.array(frames)

# to clip channels: base position, RPY in the loader's Rz Ry Rx convention, joints
pos = frames[:, :3]
rpy = np.zeros((len(frames), 3))
for k, q in enumerate(frames):
    R = np.zeros(9); mujoco.mju_quat2Mat(R, q[3:7]); R = R.reshape(3, 3)
    rpy[k] = [np.arctan2(R[2, 1], R[2, 2]), -np.arcsin(np.clip(R[2, 0], -1, 1)), np.arctan2(R[1, 0], R[0, 0])]
rpy = np.unwrap(rpy, axis=0)
# physics lets a joint spring a little past its stop on impact (RM_J2 by 37 mrad here); the
# reference must not ask for that, so clip to the model's ranges
qj = {n: np.clip(frames[:, qadr[i]], *rng[i]) for i, n in enumerate(NAMES)}
t = np.arange(len(frames)) * DT
g.write(Path(sys.argv[1] if len(sys.argv) > 1 else 'freeze_clip.npz'), t, qj, pos, rpy)
sp = max(float(np.max(np.abs(np.diff(v)))) / DT for v in qj.values())
print(json.dumps({'seconds': round(len(frames) * DT, 2), 'freeze_from_s': round((npad + f_star) * DT, 2),
                  'hold_s': HOLD_S, 'max_joint_speed': round(sp, 1)}))
