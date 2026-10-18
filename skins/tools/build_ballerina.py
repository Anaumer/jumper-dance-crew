"""Build the Ballerina upper shell: stock shell + plate tutu + bow, painted 3MF, limb colors.

Outputs (CAD millimeters, Z up, x = robot left, -y = robot front):
  work/shell.stl, work/ams.3mf, work/visual-overrides.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
import bambu3mf
from common import original_shell_cad_mm, PROFILE

OUT = Path(__file__).resolve().parents[1] / 'ballerina/work'
PALETTE = ['#F4B6C8', '#FFFFFF', '#D4AF37', '#FBD9E3']   # shell, tutu, bow, tutu blush
LIMBS = {'lower_shell': '#E7A3B7', 'primary': '#FFF3F6', 'secondary': '#F4B6C8',
         'accent': '#E8A0B6'}

# Front sector left open for the display module (x +-54 mm, y < -80 mm) and camera.
FRONT_GAP_HALF_WIDTH_MM = 1000.0   # v4: the whole front is open -- the claws rise through it (measured: v3 cut them at |x| 62-127 mm)
FRONT_GAP_Y_MM = -72.0
N_THETA = 720
CENTER_Y = -5.0
# (z of inner edge, flare beyond the shell, droop at the outer edge, slot)
LAYERS = [(66.0, 24.0, 0.5, 1), (63.5, 30.0, 1.0, 3), (61.0, 36.0, 1.5, 1), (58.5, 42.0, 2.0, 3)]
THICKNESS = 1.4
RUFFLES = 36


def shell_radius(shell, z, thetas):
    """Distance from (0, CENTER_Y) to the shell outline at height z along each direction."""
    origins = np.column_stack([np.zeros_like(thetas), np.full_like(thetas, CENTER_Y),
                               np.full_like(thetas, z)])
    dirs = np.column_stack([np.cos(thetas), np.sin(thetas), np.zeros_like(thetas)])
    locs, rays, _ = shell.ray.intersects_location(origins, dirs, multiple_hits=True)
    r = np.full(len(thetas), np.nan)
    for i in range(len(thetas)):
        hits = locs[rays == i]
        if len(hits):
            r[i] = np.max(np.linalg.norm(hits[:, :2] - [0, CENTER_Y], axis=1))
    # fill gaps from the nearest valid neighbours (open edges of the stock mesh)
    ok = ~np.isnan(r)
    r[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), r[ok], period=len(r))
    return r


def tutu_layer(shell, z0, flare, droop, phase):
    thetas = np.linspace(0, 2 * np.pi, N_THETA, endpoint=False)
    r_in = shell_radius(shell, z0, thetas) - 2.0          # tucked 2 mm into the shell
    x_out = np.cos(thetas) * (r_in + 2.0 + flare)
    y_out = CENTER_Y + np.sin(thetas) * (r_in + 2.0 + flare)
    keep = ~((np.abs(x_out) < FRONT_GAP_HALF_WIDTH_MM) & (y_out < FRONT_GAP_Y_MM))
    # longest contiguous kept run (the ring is cut once, at the front)
    idx = np.flatnonzero(keep)
    start = idx[np.argmax(np.diff(np.concatenate([idx, [idx[0] + N_THETA]])))] + 1
    order = (np.arange(N_THETA) + start) % N_THETA
    order = order[keep[order]]
    th, ri = thetas[order], r_in[order]
    rows = 6                                              # radial samples inner -> outer
    s = np.linspace(0, 1, rows)
    verts_top, verts_bot = [], []
    for t, r in zip(th, ri):
        for u in s:
            rr = r + u * (2.0 + flare) + 2.5 * u * np.sin(RUFFLES * t + phase)
            zz = z0 - droop * u ** 1.5 + 3.2 * u ** 1.2 * np.sin(RUFFLES * t + phase + np.pi / 2)
            p = [np.cos(t) * rr, CENTER_Y + np.sin(t) * rr]
            verts_top.append([*p, zz + THICKNESS / 2])
            verts_bot.append([*p, zz - THICKNESS / 2])
    n = len(th)
    V = np.array(verts_top + verts_bot)
    off = n * rows
    F = []
    vid = lambda i, j: i * rows + j
    for i in range(n - 1):
        for j in range(rows - 1):
            a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            F += [[a, b, c], [a, c, d], [off + a, off + c, off + b], [off + a, off + d, off + c]]
        for j in (0, rows - 1):                           # inner and outer rims
            a, b = vid(i, j), vid(i + 1, j)
            quad = [[a, off + a, off + b], [a, off + b, b]] if j == rows - 1 else \
                   [[a, off + b, off + a], [a, b, off + b]]
            F += quad
    for i in (0, n - 1):                                  # close both cut ends
        for j in range(rows - 1):
            a, d = vid(i, j), vid(i, j + 1)
            F += [[a, d, off + d], [a, off + d, off + a]] if i == 0 else \
                 [[a, off + d, d], [a, off + a, off + d]]
    mesh = trimesh.Trimesh(V, np.array(F), process=False)
    mesh.fix_normals()
    return mesh


def top_z(shell, x, y):
    loc, _, _ = shell.ray.intersects_location([[x, y, 200.0]], [[0, 0, -1.0]])
    return float(loc[:, 2].max())


def ellipsoid(center, radii, rotation=None):
    e = trimesh.creation.icosphere(subdivisions=3)
    e.vertices *= radii
    if rotation is not None:
        e.vertices = e.vertices @ rotation.T
    e.vertices += center
    return e


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def bow(shell):
    # behind the display, on the crown of the shell; loops stand up and lean back
    bx, by = 0.0, -38.0
    z = top_z(shell, bx, by) + 7.0
    tilt = rot_x(-1.1)                                               # stand the bow up
    parts = [ellipsoid([bx, by, z], [7, 6, 6.5])]                    # knot
    for side in (-1, 1):
        parts.append(ellipsoid([bx + side * 17, by, z + 2], [16, 5, 10],
                               tilt @ rot_y(side * 0.3)))            # loops
        parts.append(ellipsoid([bx + side * 8, by + 4, z - 9], [3.2, 2, 13],
                               rot_y(side * -0.45)))                 # tails hanging down
    return trimesh.util.concatenate(parts)


def limb_overrides():
    import xml.etree.ElementTree as ET
    prof = json.loads(PROFILE.read_text(encoding='utf-8'))
    baseline = ET.parse(PROFILE.parent / prof['baseline_mjcf']).getroot()
    hexrgba = lambda c: [int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)] + [1.0]
    records = []
    for body in baseline.iter('body'):
        name = body.get('name', '')
        if name == 'base_link':
            color = LIMBS['lower_shell']
        elif name.startswith(('LF_', 'RF_', 'LM_', 'RM_', 'LR_', 'RR_')):
            role = ('accent' if any(p in name for p in ('pad_', 'tip_', 'insert_'))
                    else 'secondary' if any(p in name for p in ('forearm_', 'thigh_'))
                    else 'primary')
            color = LIMBS[role]
        else:
            continue
        for geom in body.findall('geom'):
            if geom.get('contype') == '0' and geom.get('conaffinity') == '0':
                records.append(dict(link=name, geom=geom.get('name'), rgba=hexrgba(color)))
    return records


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    shell, _ = original_shell_cad_mm()
    pieces = [(shell, 0)]
    for k, (z0, flare, droop, slot) in enumerate(LAYERS):
        pieces.append((tutu_layer(shell, z0, flare, droop, phase=k * 1.3), slot))
    pieces.append((bow(shell), 2))
    combined = trimesh.util.concatenate([p for p, _ in pieces])
    labels = np.concatenate([np.full(len(p.faces), s, dtype=np.uint8) for p, s in pieces])
    stl = OUT / 'shell.stl'
    combined.export(stl)
    # Re-read exactly like shellflow.assemble does, then carry labels across by face center.
    reread = trimesh.load_mesh(stl, process=True)
    tree = cKDTree(combined.triangles_center)
    _, nearest = tree.query(reread.triangles_center)
    slots = labels[nearest]
    bambu3mf.write(OUT / 'ams.3mf', reread, slots, PALETTE)
    (OUT / 'visual-overrides.json').write_text(json.dumps(limb_overrides(), indent=1))
    print(json.dumps({'faces': len(reread.faces), 'bounds_mm': np.round(reread.bounds, 1).tolist(),
                      'faces_per_slot': np.bincount(slots, minlength=4).tolist()}))


if __name__ == '__main__':
    main()
