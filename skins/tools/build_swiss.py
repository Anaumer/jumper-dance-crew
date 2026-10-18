"""Build the Swiss upper shell: stock shell in Swiss red with a white Swiss cross on the crown.

The cross follows the flag's official proportions: each arm is one sixth longer than it is wide
(arm width 6, arm length 7 from the centre square -> span 20 units of a 32-unit flag). It is a
0.6 mm raised decal laid on the crown by ray casting, its outline exact because the sampling
grid is aligned to the cross edges.

Outputs (CAD millimeters): work/shell.stl, work/ams.3mf, work/visual-overrides.json
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

OUT = Path(__file__).resolve().parents[1] / 'swiss/work'
PALETTE = ['#DA291C', '#FFFFFF']
LIMBS = {'lower_shell': '#DA291C', 'primary': '#DA291C', 'secondary': '#C42418', 'accent': '#B01F15'}   # all red; slightly darker shades keep the joints readable
FLAG = 104.0               # side of the (virtual) flag square on the crown, mm
CENTER = (0.0, -28.0)      # x, y of the cross centre on the crown (front is -y; display at y < -80)
LIFT = 0.6                 # decal height above the shell surface, mm
CELLS = 40                 # grid cells across the cross span (arms are 12 cells wide)


def cross(shell):
    span, arm = FLAG * 20 / 32, FLAG * 6 / 32
    n = CELLS + 1
    g = np.linspace(-span / 2, span / 2, n)
    X, Y = np.meshgrid(g + CENTER[0], g + CENTER[1], indexing='ij')
    o = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, 200.0)])
    locs, rays, _ = shell.ray.intersects_location(o, np.tile([0, 0, -1.0], (X.size, 1)))
    top = np.full(X.size, np.nan)
    for loc, r in zip(locs, rays):
        top[r] = loc[2] if np.isnan(top[r]) else max(top[r], loc[2])
    if np.isnan(top).any():
        raise ValueError('cross does not fit on the crown: move CENTER or shrink FLAG')
    V = np.column_stack([X.ravel(), Y.ravel(), top + LIFT])
    idx = np.arange(X.size).reshape(n, n)
    cell = span / CELLS
    F = []
    for i in range(n - 1):
        for j in range(n - 1):
            cx, cy = (g[i] + g[i + 1]) / 2, (g[j] + g[j + 1]) / 2
            if abs(cx) < arm / 2 or abs(cy) < arm / 2:      # inside the vertical or horizontal bar
                a, b, c, d = idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]
                F += [[a, b, c], [a, c, d]]
    assert abs(arm / cell - 12) < 1e-9, 'grid must align with the arm edges'
    mesh = trimesh.Trimesh(V, np.array(F), process=True)
    mesh.remove_unreferenced_vertices()
    return mesh


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    shell, _ = original_shell_cad_mm()
    deco = cross(shell)
    combined = trimesh.util.concatenate([shell, deco])
    labels = np.concatenate([np.zeros(len(shell.faces), np.uint8), np.ones(len(deco.faces), np.uint8)])
    stl = OUT / 'shell.stl'
    combined.export(stl)
    reread = trimesh.load_mesh(stl, process=True)
    _, nearest = cKDTree(combined.triangles_center).query(reread.triangles_center)
    bambu3mf.write(OUT / 'ams.3mf', reread, labels[nearest], PALETTE)
    bb.LIMBS = LIMBS
    (OUT / 'visual-overrides.json').write_text(json.dumps(bb.limb_overrides(), indent=1))
    print(json.dumps({'faces': len(reread.faces), 'cross_faces': int(labels[nearest].sum())}))


if __name__ == '__main__':
    main()
