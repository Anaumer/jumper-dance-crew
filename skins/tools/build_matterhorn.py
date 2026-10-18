"""Build the Matterhorn upper shell: stock shell + a stylized Matterhorn peak, painted 3MF, limb colors.

The peak is a heightfield standing on the shell's crown: four faces meeting in ridges (an L1
"diamond" pyramid turned 45 degrees), the summit hooked over towards the robot's front-left the
way it leans in the classic Zermatt view, rock facets from a few octaves of value noise, and snow
where the surface is high and not too steep. The display module and camera at the front stay clear.

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
import build_ballerina as bb

OUT = Path(__file__).resolve().parents[1] / 'matterhorn/work'
PALETTE = ['#4F7F3A', '#7A6E62', '#F4F6F8', '#5E5348']     # meadow shell, rock, snow, dark rock
LIMBS = {'lower_shell': '#3E6630', 'primary': '#F2F2F2', 'secondary': '#D52B1E',
         'accent': '#4A4A4A'}
CENTER = (0.0, 8.0)        # x, y of the peak's footprint centre on the crown
RADIUS = 56.0              # footprint half-diagonal
HEIGHT = 84.0              # summit above the crown
HOOK = (-15.0, -7.0)       # summit offset (left, front) reached at the top
SNOWLINE = 0.55            # fraction of HEIGHT above which gentle faces hold snow
STEP = 1.5                 # grid spacing, mm


def value_noise(x, y, seed, scale):
    rng = np.random.default_rng(seed)
    grid = rng.standard_normal((64, 64))
    gx, gy = x / scale + 32, y / scale + 32
    i, j = np.floor(gx).astype(int), np.floor(gy).astype(int)
    fx, fy = gx - i, gy - j
    sx, sy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    g = lambda a, b: grid[np.clip(a, 0, 63), np.clip(b, 0, 63)]
    top = g(i, j) * (1 - sx) + g(i + 1, j) * sx
    bot = g(i, j + 1) * (1 - sx) + g(i + 1, j + 1) * sx
    return top * (1 - sy) + bot * sy


def peak(shell):
    n = int(2 * RADIUS / STEP) + 1
    xs = np.linspace(CENTER[0] - RADIUS, CENTER[0] + RADIUS, n)
    ys = np.linspace(CENTER[1] - RADIUS, CENTER[1] + RADIUS, n)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    u, v = X - CENTER[0], Y - CENTER[1]
    # diamond footprint: ridges run to the four corners (front, back, left, right)
    r = (np.abs(u) + np.abs(v)) / RADIUS
    base = np.clip(1 - r, 0, None)
    # concave, steep faces: the Matterhorn narrows fast and then stands up as a horn
    h = HEIGHT * base ** 1.7
    # the Hornli shoulder: a step on the right-hand ridge at about a third of the height
    h += 13.0 * np.exp(-((u - 20.0) ** 2 + (v + 6.0) ** 2) / 120.0) * (base > 0.25)
    # facets: rock texture fading out at the footprint edge
    rough = (value_noise(u, v, 1, 9.0) * 2.2 + value_noise(u, v, 2, 4.0) * 1.0)
    h = np.where(base > 0, h + rough * np.clip(base * 3, 0, 1), 0)
    h = np.clip(h, 0, None)
    # hook: shift the upper part of the peak towards HOOK, quadratically with height
    t = (h / HEIGHT) ** 2
    Xs, Ys = X + HOOK[0] * t, Y + HOOK[1] * t
    # stand it on the crown, sunk 1.5 mm so no seam shows
    origins = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, 200.0)])
    locs, rays, _ = shell.ray.intersects_location(origins, np.tile([0, 0, -1.0], (X.size, 1)))
    crown = np.full(X.size, np.nan)
    for loc, ray in zip(locs, rays):
        crown[ray] = loc[2] if np.isnan(crown[ray]) else max(crown[ray], loc[2])
    crown = crown.reshape(X.shape)
    keep = ~np.isnan(crown) & (base > 0)
    Z = crown - 1.5 + h
    V = np.column_stack([Xs.ravel(), Ys.ravel(), Z.ravel()])
    idx = np.arange(X.size).reshape(X.shape)
    F = []
    for i in range(n - 1):
        for j in range(n - 1):
            if keep[i, j] and keep[i + 1, j] and keep[i, j + 1] and keep[i + 1, j + 1]:
                a, b, c, d = idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]
                F += [[a, b, c], [a, c, d]]
    mesh = trimesh.Trimesh(V, np.array(F), process=True)
    mesh.remove_unreferenced_vertices()
    # colours: snow on high, not-too-steep faces; dark rock on the steepest
    centers = mesh.triangles_center
    hz = (centers[:, 2] - np.nanmin(crown[keep])) / HEIGHT
    up = mesh.face_normals[:, 2]
    slots = np.full(len(mesh.faces), 1, dtype=np.uint8)
    slots[up < 0.45] = 3
    snow_line = SNOWLINE - 0.12 * value_noise(centers[:, 0], centers[:, 1], 3, 6.0)
    slots[(hz > snow_line) & (up > 0.30)] = 2
    slots[hz > 0.86] = 2
    return mesh, slots


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    shell, _ = original_shell_cad_mm()
    mountain, mslots = peak(shell)
    combined = trimesh.util.concatenate([shell, mountain])
    labels = np.concatenate([np.zeros(len(shell.faces), np.uint8), mslots])
    stl = OUT / 'shell.stl'
    combined.export(stl)
    reread = trimesh.load_mesh(stl, process=True)
    _, nearest = cKDTree(combined.triangles_center).query(reread.triangles_center)
    slots = labels[nearest]
    bambu3mf.write(OUT / 'ams.3mf', reread, slots, PALETTE)
    bb.LIMBS = LIMBS
    (OUT / 'visual-overrides.json').write_text(json.dumps(bb.limb_overrides(), indent=1))
    print(json.dumps({'faces': len(reread.faces), 'bounds_mm': np.round(reread.bounds, 1).tolist(),
                      'faces_per_slot': np.bincount(slots, minlength=4).tolist()}))


if __name__ == '__main__':
    main()
