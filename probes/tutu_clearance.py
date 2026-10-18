"""Do any claws or legs touch or pass through the Ballerina tutu?

The tutu is checked as the solid envelope its four ruffled layers sweep (build_ballerina.py):
around the shell outline, from 2 mm inside it to the outermost flare plus the ruffle
amplitude, between the lowest and highest point any layer reaches, minus the front gap left
for the display. A limb mesh vertex inside that envelope is a contact -- conservative, since
the layers are thinner than their envelope.

usage: tutu_clearance.py clip <clip.npz>         (a dance clip, its meas_* channels)
       tutu_clearance.py states <states.npz>     (qpos [T, nq] recorded from a replay)
"""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROBOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROBOT / 'skins/tools'))
sys.path.insert(0, str(ROBOT / 'choreo'))
import build_ballerina as bb  # noqa: E402
from common import original_shell_cad_mm  # noqa: E402

shell, prof = original_shell_cad_mm()
T = prof['cad_mm_to_parent_m']
R, t0, scale = np.asarray(T['rotation_matrix']), np.asarray(T['translation_m']), T['scale']
thetas = np.linspace(0, 2 * np.pi, 720, endpoint=False)
r_shell = bb.shell_radius(shell, 62.0, thetas)
r_out = max(f for _, f, _, _ in bb.LAYERS) + 2.0 + 2.5 + 1.0          # flare + tuck + ruffle + sheet
z_lo = min(z0 - d for z0, _, d, _ in bb.LAYERS) - 3.2 - bb.THICKNESS
z_hi = max(z0 for z0, _, _, _ in bb.LAYERS) + 3.2 + bb.THICKNESS

m = mujoco.MjModel.from_xml_path(str(ROBOT / 'skins/ballerina/train-model/jumper_ballerina_v4.xml'))
d = mujoco.MjData(m)
B = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
upper = B('upper_shell_link')
ROOTS = {'LF': 'LF_shoulder_link', 'RF': 'RF_shoulder_link', 'LM': 'LM_hip_link',
         'RM': 'RM_hip_link', 'LR': 'LR_hip_link', 'RR': 'RR_hip_link'}
limb_geoms = {k: [] for k in ROOTS}
for gi in range(m.ngeom):
    if m.geom_type[gi] != 7 or m.geom_group[gi] != 2:
        continue
    b = m.geom_bodyid[gi]
    while b > 0:
        for k, root in ROOTS.items():
            if b == B(root):
                limb_geoms[k].append(gi)
        b = m.body_parentid[b]


def verts(gi):
    mid = m.geom_dataid[gi]
    v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    return v[::4]


VERTS = {gi: verts(gi) for k in limb_geoms for gi in limb_geoms[k]}


def inside_count(limb):
    """Vertices of this limb inside the tutu envelope, and the deepest one's depth (mm)."""
    Ru, pu = d.xmat[upper].reshape(3, 3), d.xpos[upper]
    pts = []
    for gi in limb_geoms[limb]:
        w = VERTS[gi] @ d.geom_xmat[gi].reshape(3, 3).T + d.geom_xpos[gi]
        pts.append(w)
    w = np.concatenate(pts)
    parent = (w - pu) @ Ru                       # world -> upper_shell_link frame
    cad = ((parent - t0) @ R) / scale            # -> CAD millimetres
    x, y, z = cad[:, 0], cad[:, 1] - bb.CENTER_Y, cad[:, 2]
    th = np.mod(np.arctan2(y, x), 2 * np.pi)
    r = np.hypot(x, y)
    rs = np.interp(th, thetas, r_shell, period=2 * np.pi)
    gap = (np.abs(cad[:, 0]) < bb.FRONT_GAP_HALF_WIDTH_MM) & (cad[:, 1] < bb.FRONT_GAP_Y_MM)
    inside = (r > rs - 2.0) & (r < rs + r_out) & (z > z_lo) & (z < z_hi) & ~gap
    if not inside.any():
        return 0, 0.0
    depth = np.minimum.reduce([r[inside] - (rs[inside] - 2.0), (rs[inside] + r_out) - r[inside],
                               z[inside] - z_lo, z_hi - z[inside]])
    return int(inside.sum()), float(depth.max())


def qpos_from_clip(path):
    import generate as g
    c = np.load(path)
    J = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j): j for j in range(m.njnt)}
    n = len(c['time'])
    Q = np.zeros((n, m.nq))
    Q[:, 0], Q[:, 1], Q[:, 2] = c['meas_body_x'], c['meas_body_y'], c['meas_body_z']
    rpy = np.stack([c['meas_body_roll'], c['meas_body_pitch'], c['meas_body_yaw']], 1)
    Q[:, 3:7] = g._rpy_to_quat(rpy)
    for leg, js in g.LEG_JOINTS.items():
        for j, nm in enumerate(js):
            Q[:, m.jnt_qposadr[J[nm]]] = c[f'meas_L{leg}_j{j}']
    return Q, float(c['dt'])


if __name__ == '__main__':
    mode, path = sys.argv[1], sys.argv[2]
    if mode == 'clip':
        Q, dt = qpos_from_clip(path)
    else:
        s = np.load(path); Q, dt = s['qpos'], float(s['dt'])
    step = max(1, int(round(0.04 / dt)))
    hits = {k: [] for k in ROOTS}
    for i in range(0, len(Q), step):
        d.qpos[:] = Q[i]
        mujoco.mj_forward(m, d)
        for k in ROOTS:
            cnt, depth = inside_count(k)
            if cnt:
                hits[k].append((round(i * dt, 2), cnt, round(depth, 1)))
    summary = {}
    for k, h in hits.items():
        if h:
            summary[k] = {'frames_touching': len(h), 'first_t': h[0][0], 'last_t': h[-1][0],
                          'max_depth_mm': max(x[2] for x in h)}
    print(json.dumps({'checked_frames': len(range(0, len(Q), step)), 'touching': summary or 'none'}, indent=1))
    if summary:
        # contiguous windows, for reading against the choreography
        for k, h in hits.items():
            if not h:
                continue
            ts = [x[0] for x in h]; wins = [[ts[0], ts[0]]]
            for a in ts[1:]:
                if a - wins[-1][1] > 0.1:
                    wins.append([a, a])
                else:
                    wins[-1][1] = a
            print(k, 'touches during', [tuple(w) for w in wins][:12])
