"""Build the China upper shell: all red, the PRC flag's five yellow stars on the crown.

Geometry follows the flag's construction sheet on its 30 x 20 grid: the large star (radius 3)
centred at (5, 5); four small stars (radius 1) at (10, 2), (12, 4), (12, 7), (10, 9), each
turned so one point aims at the large star's centre. The flag rectangle is laid on the crown
as seen from above with the robot's front at the top; the stars are 0.6 mm raised decals,
triangulated from their exact outline and projected onto the shell.
"""
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
import bambu3mf
import build_ballerina as bb
from common import original_shell_cad_mm

OUT = Path(__file__).resolve().parents[1] / 'china/work'
PALETTE = ['#EE1C25', '#FFDE00']
LIMBS = {'lower_shell': '#EE1C25', 'primary': '#EE1C25', 'secondary': '#D41A21', 'accent': '#B8161C'}
FLAG_H = 125.0             # flag height on the crown, mm (width 1.5 x); the stars use its left third
FLAG_CENTER = (-8.0, -16.0) # CAD x, y of the flag centre
LIFT = 0.6
UNIT = FLAG_H / 20.0
STARS = [((5, 5), 3.0), ((10, 2), 1.0), ((12, 4), 1.0), ((12, 7), 1.0), ((10, 9), 1.0)]


def flag_to_cad(u, v):
    """Flag grid (u right, v down, origin top-left) -> CAD mm. Top of the flag = robot front
    (-y); seen from above with the front at the top, the flag's left edge is the robot's left (+x)."""
    x = FLAG_CENTER[0] + (15.0 - u) * UNIT
    y = FLAG_CENTER[1] + (v - 10.0) * UNIT
    return x, y


def star_outline(center, radius, point_at=None):
    cu, cv = center
    if point_at is None:
        a0 = -np.pi / 2                                   # large star: a point straight up
    else:
        a0 = np.arctan2(point_at[1] - cv, point_at[0] - cu)
    pts = []
    for k in range(10):
        r = radius if k % 2 == 0 else radius * 0.381966   # regular five-point star
        a = a0 + k * np.pi / 5
        pts.append((cu + r * np.cos(a), cv + r * np.sin(a)))
    return pts


def star_mesh(shell, center, radius, point_at=None):
    outline = star_outline(center, radius, point_at)
    uv = np.array([center] + outline)
    xy = np.array([flag_to_cad(u, v) for u, v in uv])
    faces = [[0, 1 + k, 1 + (k + 1) % 10] for k in range(10)]
    flat = trimesh.Trimesh(np.c_[xy, np.zeros(len(xy))], faces, process=False)
    v, f = trimesh.remesh.subdivide_to_size(flat.vertices, flat.faces, max_edge=1.5)
    o = np.c_[v[:, :2], np.full(len(v), 200.0)]
    locs, rays, _ = shell.ray.intersects_location(o, np.tile([0, 0, -1.0], (len(v), 1)))
    top = np.full(len(v), np.nan)
    for loc, r in zip(locs, rays):
        top[r] = loc[2] if np.isnan(top[r]) else max(top[r], loc[2])
    if np.isnan(top).any():
        raise ValueError('a star falls off the crown: move FLAG_CENTER or shrink FLAG_H')
    m = trimesh.Trimesh(np.c_[v[:, :2], top + LIFT], f, process=True)
    if m.face_normals[:, 2].mean() < 0:
        m.invert()
    return m


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    shell, _ = original_shell_cad_mm()
    stars = [star_mesh(shell, *STARS[0])] + [star_mesh(shell, c, r, point_at=STARS[0][0]) for c, r in STARS[1:]]
    deco = trimesh.util.concatenate(stars)
    combined = trimesh.util.concatenate([shell, deco])
    labels = np.concatenate([np.zeros(len(shell.faces), np.uint8), np.ones(len(deco.faces), np.uint8)])
    stl = OUT / 'shell.stl'
    combined.export(stl)
    reread = trimesh.load_mesh(stl, process=True)
    _, nearest = cKDTree(combined.triangles_center).query(reread.triangles_center)
    bambu3mf.write(OUT / 'ams.3mf', reread, labels[nearest], PALETTE)
    bb.LIMBS = LIMBS
    (OUT / 'visual-overrides.json').write_text(json.dumps(bb.limb_overrides(), indent=1))
    print(json.dumps({'faces': len(reread.faces), 'star_faces': int(labels[nearest].sum()),
                      'large_star_centre_cad': [round(c, 1) for c in flag_to_cad(5, 5)]}))


if __name__ == '__main__':
    main()
