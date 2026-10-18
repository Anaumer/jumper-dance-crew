"""B-boy battle v3, cut to Musik/breakdance2.mp3: the crabs stay close, the active one dances
round the other on a half circle (a courtship dance), the waiting one never takes its eyes off it.

The track (measured from its onsets): 90.0 BPM, beat k at 0.04 + k * 0.6667 s; the drop is beat
15 (10.04 s), the loud part ends at 48.0 s (beat 72), the track at 59.6 s.
  intro   0-10 s   both uprock, walk up, fist bump with the claws, step back
  round A 10-32 s  camo: purzelbaum on the drop, half circle, radschlag, freeze, pinch, spin away
  round B 32-48 s  black: uprock while camo spins away, radschlag, headspin
  outro   48-59 s  banter only: pinch, a second fist bump, both turn to the camera

Every frame is physics, from one of four sources:
  loco      the shipped locomotion policy (loco.py), driven closed-loop in the stage frame:
            'watch' turns to face the other crab and keeps a distance, 'orbit' walks sideways
            round the other crab facing it, 'spin' turns a full 360 away and back. All bob to
            the beat (a dip and a nod on every beat).
  gestures  replay of jumper.battle_gestures (uprock, pinch, applause + covering the eyes);
            the fist bump is the pinch for both crabs at once, the second one mirrored, placed
            mirror-symmetrically so the two claws meet in the plane between them
  moves     open-loop physics moves (radschlag, purzelbaum, headspin), chain / headspin_search
  freeze    replay of jumper.freeze
Segments are placed rigidly where the last one ended and cross-faded (0.1 s; 0.3 s into a fist
bump, which also absorbs the up-to-few-cm correction to the exact bump distance).

usage: battle3.py <gesture_replay.npz> <freeze_replay.npz> <out.npz>
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

import battle_stage as bs
import chain
import loco
import roll_search as rs

DT = 0.01
BEAT = 60 / 90.01
OFF = 0.04                     # time of beat 0
DROP, LOUD_END, TRACK_END = 15, 72, 59.6
R = 0.55                       # distance between the crabs while one dances round the other
HERE = Path(__file__).parent


def beat_t(k):
    return OFF + k * BEAT


def next_beat(t, plus=0):
    # timelines grow in whole frames, so a line 'on' a beat may sit up to one frame past it
    return beat_t(np.ceil((t - DT * 1.1 - OFF) / BEAT) + plus)


def resample(Q, dt):
    if abs(dt - DT) < 1e-9:
        return Q
    idx = np.round(np.arange(0, len(Q) * dt, DT) / dt).astype(int)
    return Q[np.minimum(idx, len(Q) - 1)]


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def rot(yaw, v):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


# ── the Train model: joint addresses, mirroring, the claw's reach ───────────────────────────
TM = mujoco.MjModel.from_xml_path(str(rs.ROBOT / 'jumper/assets/jumper/jumper.xml'))
ADR = {n: TM.jnt_qposadr[mujoco.mj_name2id(TM, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in rs.HOME}


def mirror(Q):
    """Left-right mirror image of qpos rows (reflection y -> -y; chain.mirror's joint rule)."""
    Q = np.asarray(Q); M = Q.copy()
    M[:, 1] = -Q[:, 1]
    M[:, 4] = -Q[:, 4]; M[:, 6] = -Q[:, 6]                 # quat (w, x, y, z) -> (w, -x, y, -z)
    for n, a in ADR.items():
        other = ('R' if n[0] == 'L' else 'L') + n[1:]
        k = int(n[4])
        M[:, ADR[other]] = (chain.ARM_SIGN[k] if n[1] == 'F' else -1) * Q[:, a]
    return M


def claw_reach(Q, side='LF'):
    """Furthest the claw gets ahead of the base, along the segment's starting heading [m]."""
    d = mujoco.MjData(TM)
    gis = [g for g in range(TM.ngeom) if TM.geom_type[g] == 7 and TM.geom_contype[g]
           and (TM.geom(g).name.startswith(side + '_palm') or TM.geom(g).name.startswith(side + '_finger'))]
    y0 = bs.yaw_of(Q[0, 3:7]); fwd = np.array([np.cos(y0), np.sin(y0), 0]); p0 = Q[0, :3]
    best = -np.inf
    for q in Q[::2]:
        d.qpos[:] = q; mujoco.mj_kinematics(TM, d)
        for g in gis:
            mid = TM.geom_dataid[g]
            v = TM.mesh_vert[TM.mesh_vertadr[mid]:TM.mesh_vertadr[mid] + TM.mesh_vertnum[mid]]
            w = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
            best = max(best, float(((w - p0) @ fwd).max()))
    return best


# ── fixed segments ────────────────────────────────────────────────────────────────────────
g = np.load(sys.argv[1]); fz = np.load(sys.argv[2])
G = resample(g['qpos'], float(g['dt'])); F = resample(fz['qpos'], float(fz['dt']))
cut = lambda Q, a, b: Q[int(round(a / DT)):int(round(b / DT))]
SEG = {'uprock': cut(G, 27.84, 31.58), 'pinch': cut(G, 15.39, 18.72),
       'clap_eyes': cut(G, 5.944, 15.39), 'freeze': cut(F, 2.4, 8.2)}   # settled from 8.2 s
for name in ('radschlag', 'purzelbaum'):
    Q, up, sp = chain.simulate([chain.Move(name, False)], [0.0], 5.0)
    end = chain.settle_time(up, sp, after=2.0) or 4.0
    SEG[name] = Q[:int((end + 0.1) / DT)]
if (HERE / 'headspin_states.npz').exists():
    SEG['headspin'] = resample(np.load(HERE / 'headspin_states.npz')['qpos'], 0.01)
SEG['uprock_m'] = mirror(SEG['uprock'])
D_BUMP = 2 * claw_reach(SEG['pinch']) + 0.001             # the claws meet (0.5 mm each side)
print(f'fist bump distance {D_BUMP:.3f} m', flush=True)


def body_disp(Q):
    y0 = bs.yaw_of(Q[0, 3:7])
    return rot(-y0, Q[-1, :2] - Q[0, :2])


RAD_D = body_disp(SEG['radschlag'])        # (-0.25, +0.79): the cartwheel travels to the left
# facing the partner, the body's left points clockwise round it: a cartwheel to the left
# travels clockwise, so the orbit does too
ORBIT = -1 if RAD_D[1] > 0 else 1


# ── a timeline is a list of qpos rows at DT; helpers append to it ────────────────────────
class Line:
    def __init__(self, xy, yaw, joints_from):
        self.Q = [bs.place(joints_from[:1].copy(), xy, yaw)[0]]

    @property
    def t(self):
        return len(self.Q) * DT

    def pose(self):
        q = self.Q[-1]
        return q[:2].copy(), bs.yaw_of(q[3:7])

    def add(self, S, blend_s=0.1, placed=False):
        if not placed:
            xy, yaw = self.pose()
            S = bs.place(np.asarray(S), xy, yaw)
        S = np.array(S)
        nb = min(int(blend_s / DT), len(S) - 1)
        last = self.Q[-1]
        for k in range(nb):
            u = (k + 1) / (nb + 1)
            S[k, :3] = (1 - u) * last[:3] + u * S[k, :3]
            S[k, 7:] = (1 - u) * last[7:] + u * S[k, 7:]
            qs = S[k, 3:7] if np.dot(last[3:7], S[k, 3:7]) >= 0 else -S[k, 3:7]   # q and -q: same turn
            qq = (1 - u) * last[3:7] + u * qs; S[k, 3:7] = qq / np.linalg.norm(qq)
        self.Q.extend(S)

    def xy_at(self, t):
        i = min(int(round(t / DT)), len(self.Q) - 1)
        return self.Q[i][:2]

    def hold_to(self, t):
        while self.t < t - DT / 2:                            # to the frame nearest t
            self.Q.append(self.Q[-1].copy())


def bob(tg):
    """Height and pitch for global time tg: a dip and a nod on every beat."""
    ph = ((tg - OFF) % BEAT) / BEAT
    dip = 0.5 * (1 + np.cos(2 * np.pi * ph))
    return loco.STAND_H - 0.012 * dip, 0.10 * dip


def loco_seg(line, dur, mode, other=None, center=None, chord_end=False, spin_dir=1, keep=R,
             vmax=0.12):
    """Simulate `dur` s of the locomotion policy from `line`'s current pose, closed-loop in the
    stage frame, and append it. `other` is the other crab's Line (read at global time)."""
    t0 = line.t
    xy0, yaw0 = line.pose()
    st = {}

    def cmd(t, xy_env, yaw_env):
        if 'T' not in st:                                     # the reset pose -> stage pose
            st['T'] = (xy_env.copy(), yaw_env); st['yaw_u'] = yaw0; st['prev'] = yaw0
        p_env, y_env0 = st['T']
        xy = xy0 + rot(yaw0 - y_env0, xy_env - p_env)
        yaw = wrap(yaw0 + yaw_env - y_env0)
        st['yaw_u'] += wrap(yaw - st['prev']); st['prev'] = yaw
        tg = t0 + t
        h, pitch = bob(tg)
        tgt = center if center is not None else other.xy_at(tg)
        to = tgt - xy; dist = np.linalg.norm(to)
        face = np.arctan2(to[1], to[0])
        if mode == 'spin':                                    # 360 away and round, then face
            T_turn = 4 * BEAT
            if t < T_turn:
                want = yaw0 + spin_dir * 2 * np.pi * (0.5 - 0.5 * np.cos(np.pi * t / T_turn))
                ff = spin_dir * np.pi ** 2 / T_turn * np.sin(np.pi * t / T_turn)
                wz = np.clip(5.0 * (want - st['yaw_u']) + ff, -4, 4)
                return (0.0, 0.0, wz, 0.0, 0.0, 0.0, loco.STAND_H)
        if mode == 'orbit':
            v = 0.25
            want = face
            if chord_end and t > dur - 1.0:                   # line the cartwheel up with a chord
                L = np.linalg.norm(RAD_D); phi = np.arcsin(min(1.0, L / (2 * R)))
                rhat = -to / max(dist, 1e-6); that = ORBIT * np.array([-rhat[1], rhat[0]])
                u = np.cos(phi) * that - np.sin(phi) * rhat
                yaw_req = np.arctan2(u[1], u[0]) - np.arctan2(RAD_D[1], RAD_D[0])
                a = min(1.0, (t - (dur - 1.0)) / 0.6)
                want = face + a * wrap(yaw_req - face)
                v *= (1 - a)
            err = wrap(want - yaw)
            vy = -ORBIT * v if abs(err) < 0.6 else 0.0
            vx = np.clip(1.5 * (dist - R), -0.15, 0.15)
            wz = np.clip(4.0 * err + ORBIT * v / R, -3.0, 3.0)
            return (vx, vy, wz, 0.0, pitch, 0.0, h)
        err = wrap(face - yaw)                                # 'watch' / after the spin
        vx = np.clip(1.5 * (dist - keep), -vmax, vmax) if abs(err) < 0.5 else 0.0
        wz = np.clip(4.0 * err, -3.0, 3.0)
        if abs(err) < 0.05:
            wz = 0.0
        return (vx, 0.0, wz, 0.0, pitch, 0.0, h)

    r = loco.run(cmd, dur + DT)
    n = int(round((t0 + dur) / DT)) - len(line.Q)              # end on the frame nearest t0 + dur
    line.add(resample(r['qpos'], r['dt'])[:n])


def watch_until(line, other, t_end, center=None, keep=R, vmax=0.12):
    if t_end - line.t > 0.05:
        loco_seg(line, t_end - line.t, 'watch', other=other, center=center, keep=keep, vmax=vmax)


def fist_bump(a, b):
    """Both crabs at once, on the next beat: the pinch, b's mirrored, placed mirror-symmetric
    about the plane between them at the bump distance, facing each other exactly."""
    t = next_beat(max(a.t, b.t)); a.hold_to(t); b.hold_to(t)
    pa, _ = a.pose(); pb, _ = b.pose()
    u = (pb - pa) / np.linalg.norm(pb - pa); mid = 0.5 * (pa + pb); yu = float(np.arctan2(u[1], u[0]))
    a.add(bs.place(SEG['pinch'], mid - u * D_BUMP / 2, yu), blend_s=0.3, placed=True)
    b.add(bs.place(mirror(SEG['pinch']), mid + u * D_BUMP / 2, yu + np.pi), blend_s=0.3, placed=True)
    print(f'  fist bump {t:6.2f}-{a.t:6.2f} s (moved {np.linalg.norm(pb - pa) - D_BUMP:+.3f} m)', flush=True)


def approach(a, b, beats, keep):
    """Walk up to each other: a goes half way toward b's spot, then b closes up to a."""
    t = next_beat(max(a.t, b.t)); a.hold_to(t); b.hold_to(t)
    pa, _ = a.pose(); pb, _ = b.pose(); d0 = np.linalg.norm(pb - pa)
    # aim 2 cm short of the bump distance: the distance loop is proportional and settles short
    loco_seg(a, beats * BEAT, 'watch', center=pb, keep=d0 - (d0 - keep) / 2 - 0.01, vmax=0.2)
    loco_seg(b, beats * BEAT, 'watch', other=a, keep=keep - 0.02, vmax=0.2)


def step(line, name, accents):
    line.hold_to(next_beat(line.t)); accents.append(int(round((line.t - OFF) / BEAT)))
    t = line.t
    line.add(SEG[name])
    print(f'  {name:10s} {t:6.2f}-{line.t:6.2f} s', flush=True)
    return t


# ── the battle ───────────────────────────────────────────────────────────────────────────
# Orbits run through the far side of the circle (+y, away from the camera at -y), so the
# active crab faces the camera while it faces the other one. ORBIT=-1 (clockwise): start west
# of the partner (angle pi) and go round through +y.
start_angle = 0.0 if ORBIT > 0 else np.pi
R_INTRO = 0.65                 # the uprock walks each crab ~6 cm forward: start further apart
B0 = np.array([R_INTRO / 2, 0.0]) * (-1 if ORBIT > 0 else 1)
A0 = B0 + R_INTRO * np.array([np.cos(start_angle), np.sin(start_angle)])
face = lambda p, q: float(np.arctan2(q[1] - p[1], q[0] - p[0]))
A = Line(A0, face(A0, B0), SEG['uprock']); B = Line(B0, face(B0, A0), SEG['uprock'])
acc = []

# intro: uprock together, walk up, fist bump, step back for the drop
# not in lockstep: camo opens, black answers a beat later with the mirrored uprock
A.hold_to(beat_t(0)); step(A, 'uprock', acc)
watch_until(B, A, beat_t(1), center=A0); step(B, 'uprock_m', acc)
approach(A, B, 2, D_BUMP)
fist_bump(A, B)
A.hold_to(next_beat(A.t)); B.hold_to(next_beat(B.t))
watch_until(A, B, beat_t(DROP), center=B.pose()[0], keep=R, vmax=0.2)
watch_until(B, A, beat_t(DROP), keep=R, vmax=0.2)
assert max(A.t, B.t) <= beat_t(DROP) + 2 * DT, 'intro runs past the drop'


def orbit(line, beats, center, chord_end):
    line.hold_to(next_beat(line.t))
    loco_seg(line, beats * BEAT, 'orbit', center=center, chord_end=chord_end)


def face_seg(line, beats, center, extra=0.0):
    line.hold_to(next_beat(line.t))
    loco_seg(line, beats * BEAT, 'watch', center=center, keep=R + extra, vmax=0.2)


# round A: camo leads from the drop
c = B.pose()[0]
step(A, 'purzelbaum', acc)
orbit(A, 4, c, True)
step(A, 'radschlag', acc)
face_seg(A, 1, c, extra=0.15)                     # room for the freeze, which travels in
t_freeze = step(A, 'freeze', acc)
face_seg(A, 1, c)                                # the freeze lands turned away: face first
step(A, 'pinch', acc)
watch_until(B, A, next_beat(t_freeze))           # black watches, applauds, covers its eyes
B.add(SEG['clap_eyes'])
watch_until(B, A, A.t)

# round B: black goes into its uprock while camo spins away
t_b = A.t
loco_seg(A, 4 * BEAT, 'spin', other=B, spin_dir=ORBIT)
B.hold_to(t_b)
c = A.pose()[0]
step(B, 'uprock', acc)
orbit(B, 2, c, True)
step(B, 'radschlag', acc)
finale = 'headspin' if 'headspin' in SEG else 'freeze'
t_fin = step(B, finale, acc)
print(f'  last power move ends {B.t:.2f} s (loud part ends {beat_t(LOUD_END):.2f} s)')
watch_until(A, B, next_beat(t_fin))
A.add(SEG['clap_eyes'])

# outro: banter -- black turns back to camo and pinches, a second fist bump, both turn to the camera
face_seg(B, 2, c)
step(B, 'pinch', acc)
watch_until(A, B, B.t)
approach(A, B, 2, D_BUMP)
fist_bump(A, B)
cam = 0.5 * (A.pose()[0] + B.pose()[0]) + np.array([0.0, -3.0])
A.hold_to(next_beat(A.t)); watch_until(A, None, TRACK_END, center=cam, vmax=0.0)   # turn on the spot
B.hold_to(next_beat(B.t)); watch_until(B, A, next_beat(B.t, 1), keep=10.0, vmax=0.0)  # a last look
watch_until(B, None, TRACK_END, center=cam, vmax=0.0)

n = int(round(TRACK_END / DT))
pad = lambda L: np.array((L.Q + [L.Q[-1]] * max(0, n - len(L.Q)))[:n])
QA, QB = pad(A), pad(B)
np.savez(sys.argv[3], A=QA, B=QB, dt=DT, beat=BEAT, offset=OFF, accents=np.array(sorted(set(acc))))
dd = np.linalg.norm(QA[:, :2] - QB[:, :2], axis=1)
print(f'length {n * DT:.1f} s  orbit {ORBIT:+d}  distance min {dd.min():.2f} max {dd.max():.2f} m')
