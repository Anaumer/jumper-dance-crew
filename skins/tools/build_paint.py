"""Paint-only Jumper skins for the breakdance crew: colour on the STOCK shell, no geometry added.

The stock upper shell is subdivided (midpoint subdivision: every new vertex lies on an existing
face, so the surface is exactly the same) until no edge is longer than MAX_EDGE, then every face
gets a colour slot from the motif's painter, evaluated at the face centre in CAD millimetres
(x = robot left, -y = robot front, z up). Limbs get per-link colours through visual overrides.

usage: build_paint.py <motif>    motifs: graffiti, tracksuit, oldschool, neon, camo, camo_heavy, tracksuit3
writes skins/crew/<motif>/work/{shell.stl, ams.3mf, visual-overrides.json}
"""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
import bambu3mf
from common import original_shell_cad_mm, PROFILE

MAX_EDGE = 2.6   # MuJoCo reads at most 200k faces per STL: this keeps the whole shell near 165k
OUT = Path(__file__).resolve().parents[1] / 'crew'


def hexrgba(c):
    return [int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)] + [1.0]


# ── painters: face centres [N,3] (+ normals) -> slot index ───────────────────────────────
def graffiti(c, n):
    rng = np.random.default_rng(3)
    slot = np.zeros(len(c), np.uint8)                       # 0 white
    for k in range(26):                                     # spray blobs
        cx, cy = rng.uniform(-85, 85), rng.uniform(-100, 90)
        r = rng.uniform(5, 16); s = 1 + k % 3
        slot[(c[:, 0] - cx) ** 2 + (c[:, 1] - cy) ** 2 < r * r] = s
    # a tag: a thick wavy line across the crown
    y_line = -10 + 14 * np.sin(c[:, 0] / 14.0)
    slot[(np.abs(c[:, 1] - y_line) < 3.2) & (c[:, 2] > 55)] = 3
    return slot


def tracksuit(c, n):
    slot = np.zeros(len(c), np.uint8)                       # 0 black
    for x0 in (-14.0, 14.0):                                # two stripes front to back
        slot[np.abs(c[:, 0] - x0) < 4.0] = 1
    slot[(c[:, 2] < 40)] = 2                                # neon trim along the lower rim
    return slot


def oldschool(c, n):
    slot = np.where(c[:, 0] > 0, 0, 1).astype(np.uint8)    # 0 red (left), 1 black (right)
    slot[np.abs(c[:, 1] + 20) < 4.0] = 2                   # white band across
    chain = (np.abs(np.hypot(c[:, 0] / 1.15, c[:, 1] + 55) - 46) < 2.2) & (c[:, 1] > -90)
    beads = (np.sin(np.arctan2(c[:, 1] + 55, c[:, 0]) * 40) > 0)
    slot[chain & beads] = 3                                 # gold chain, beaded
    return slot


def neon(c, n):
    slot = np.zeros(len(c), np.uint8)                       # 0 navy
    for sx in (-32.0, 32.0):                                # two speakers on the crown
        r = np.hypot(c[:, 0] - sx, c[:, 1] + 20)
        slot[(np.abs(r - 18) < 2.6) | (r < 6)] = 1
    slot[np.abs(c[:, 1] - 30) < 3.0] = 2                    # pink stripe
    slot[np.abs(c[:, 1] - 40) < 1.6] = 2
    return slot


def camo(c, n):
    rng = np.random.default_rng(11)
    slot = np.zeros(len(c), np.uint8)
    for k in range(40):
        cx, cy, cz = rng.uniform(-90, 90), rng.uniform(-110, 100), rng.uniform(30, 80)
        rx, ry = rng.uniform(8, 22), rng.uniform(5, 14); a = rng.uniform(0, np.pi)
        dx, dy = c[:, 0] - cx, c[:, 1] - cy
        u, v = dx * np.cos(a) + dy * np.sin(a), -dx * np.sin(a) + dy * np.cos(a)
        slot[(u / rx) ** 2 + (v / ry) ** 2 + ((c[:, 2] - cz) / 30) ** 2 < 1] = 1 + k % 2
    return slot


def camo_heavy(c, n):
    """Woodland camouflage: three layers of overlapping blobs, big to small, four colours."""
    rng = np.random.default_rng(23)
    slot = np.zeros(len(c), np.uint8)                       # 0 olive base
    for layer, (count, sizes) in enumerate([(45, (14, 28)), (60, (8, 18)), (55, (4, 10))]):
        for k in range(count):
            cx, cy, cz = rng.uniform(-95, 95), rng.uniform(-112, 100), rng.uniform(28, 82)
            rx = rng.uniform(*sizes); ry = rx * rng.uniform(0.4, 0.8); a = rng.uniform(0, np.pi)
            dx, dy = c[:, 0] - cx, c[:, 1] - cy
            u, v = dx * np.cos(a) + dy * np.sin(a), -dx * np.sin(a) + dy * np.cos(a)
            wob = 1 + 0.25 * np.sin(np.arctan2(v, u) * 5 + k)          # ragged edges
            inside = (u / (rx * wob)) ** 2 + (v / (ry * wob)) ** 2 + ((c[:, 2] - cz) / 35) ** 2 < 1
            slot[inside] = [1, 2, 3][(k + layer) % 3]
    return slot


def tracksuit3(c, n):
    """The black tracksuit with a third white stripe down the middle of the crown."""
    slot = tracksuit(c, n)
    slot[np.abs(c[:, 0]) < 4.0] = 1
    return slot


def tracksuit3_alt(c, n):
    """Same, but the middle stripe narrow and neon -- reads as its own design, not as three equal stripes."""
    slot = tracksuit(c, n)
    slot[np.abs(c[:, 0]) < 2.0] = 2
    return slot


MOTIFS = {
    'graffiti': dict(paint=graffiti, palette=['#F4F4F4', '#FF3D7F', '#2EC4FF', '#7A3CFF'],
                     limbs={'LF': '#FF3D7F', 'RF': '#2EC4FF', 'LM': '#FFD23F', 'RM': '#7CFF4F',
                            'LR': '#7A3CFF', 'RR': '#FF8C1A', 'base': '#F4F4F4'}),
    'tracksuit3': dict(paint=tracksuit3, palette=['#1A1A1A', '#FFFFFF', '#B8F32A'],
                       limbs={'*': '#1A1A1A', 'tip': '#B8F32A', 'base': '#1A1A1A', 'stripe': '#FFFFFF'}),
    'tracksuit3_alt': dict(paint=tracksuit3_alt, palette=['#1A1A1A', '#FFFFFF', '#B8F32A'],
                           limbs={'*': '#1A1A1A', 'tip': '#B8F32A', 'base': '#1A1A1A', 'stripe': '#FFFFFF'}),
    'tracksuit': dict(paint=tracksuit, palette=['#1A1A1A', '#FFFFFF', '#B8F32A'],
                      limbs={'*': '#1A1A1A', 'tip': '#B8F32A', 'base': '#1A1A1A', 'stripe': '#FFFFFF'}),
    'oldschool': dict(paint=oldschool, palette=['#D62828', '#1A1A1A', '#FFFFFF', '#D4AF37'],
                      limbs={'L': '#D62828', 'R': '#1A1A1A', 'tip': '#FFFFFF', 'base': '#FFFFFF'}),
    'neon': dict(paint=neon, palette=['#1B1F4B', '#2EF2FF', '#FF2E97'],
                 limbs={'*': '#1B1F4B', 'tip': '#2EF2FF', 'stripe': '#FF2E97', 'base': '#1B1F4B'}),
    'camo_heavy': dict(paint=camo_heavy, palette=['#6B7F4A', '#3E4A2A', '#A89A6A', '#2B2A20'],
                       limbs={'camo': ['#6B7F4A', '#3E4A2A', '#A89A6A', '#2B2A20', '#55663A'],
                              'base': '#3E4A2A'}),
    'camo': dict(paint=camo, palette=['#6B7F4A', '#3E4A2A', '#A89A6A'],
                 limbs={'*': '#4F5D36', 'tip': '#A89A6A', 'stripe': '#3E4A2A', 'base': '#3E4A2A'}),
}


def limb_overrides(spec):
    prof = json.loads(PROFILE.read_text(encoding='utf-8'))
    baseline = ET.parse(PROFILE.parent / prof['baseline_mjcf']).getroot()
    L = spec['limbs']; records = []
    for body in baseline.iter('body'):
        name = body.get('name', '')
        if name == 'base_link':
            color = L['base']
        elif name[:2] in ('LF', 'RF', 'LM', 'RM', 'LR', 'RR') and 'camo' in L:
            # every link its own camouflage colour, so the legs break up like the shell
            color = L['camo'][sum(map(ord, name)) % len(L['camo'])]
        elif name[:2] in ('LF', 'RF', 'LM', 'RM', 'LR', 'RR'):
            tip = any(p in name for p in ('pad_', 'tip_', 'insert_'))
            mid = any(p in name for p in ('forearm_', 'thigh_'))
            if tip and 'tip' in L:
                color = L['tip']
            elif name[:2] in L:
                color = L[name[:2]]
            elif name[0] in L:
                color = L[name[0]]
            elif mid and 'stripe' in L:
                color = L['stripe']
            else:
                color = L.get('*', '#808080')
        else:
            continue
        for geom in body.findall('geom'):
            if geom.get('contype') == '0' and geom.get('conaffinity') == '0':
                records.append(dict(link=name, geom=geom.get('name'), rgba=hexrgba(color)))
    return records


def main(motif):
    spec = MOTIFS[motif]
    out = OUT / motif / 'work'; out.mkdir(parents=True, exist_ok=True)
    shell, _ = original_shell_cad_mm()
    v, f = trimesh.remesh.subdivide_to_size(shell.vertices, shell.faces, max_edge=spec.get('max_edge', MAX_EDGE))
    fine = trimesh.Trimesh(v, f, process=False)
    for box in spec.get('refine', []):
        for _ in range(box['levels']):
            cc, nn = fine.triangles_center, fine.face_normals
            sel = np.where((cc[:, 0] > box['x'][0]) & (cc[:, 0] < box['x'][1]) & (cc[:, 1] > box['y'][0])
                           & (cc[:, 1] < box['y'][1]) & (cc[:, 2] > box['z'][0]) & (cc[:, 2] < box['z'][1])
                           & (nn[:, 1] <= box.get('ny_max', 1.0)) & (nn[:, 2] >= box.get('nz_min', -1.0)))[0]
            fine = fine.subdivide(face_index=sel)    # midpoint split: the surface stays the stock one
    stl = out / 'shell.stl'
    fine.export(stl)
    reread = trimesh.load_mesh(stl, process=True)
    slots = spec['paint'](reread.triangles_center, reread.face_normals)
    bambu3mf.write(out / 'ams.3mf', reread, slots, spec['palette'])
    (out / 'visual-overrides.json').write_text(json.dumps(limb_overrides(spec), indent=1))
    # the surface is the stock one: every subdivided vertex lies on the stock shell
    _, dist, _ = trimesh.proximity.closest_point(shell, reread.vertices[::max(1, len(reread.vertices) // 3000)])
    print(json.dumps({'motif': motif, 'faces': len(reread.faces), 'max_dev_from_stock_mm': round(float(dist.max()), 6),
                      'faces_per_slot': np.bincount(slots, minlength=len(spec['palette'])).tolist()}))


if __name__ == '__main__':
    main(sys.argv[1])
