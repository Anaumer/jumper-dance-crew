"""Render the composed battle: both crabs on one stage, side camera following their midpoint.
usage: battle_render.py <battle.npz> <skin_a.xml> <skin_b.xml> <out.mp4>"""
import sys
import imageio
import mujoco
import numpy as np
import battle_stage as bs
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'choreo'))
import posekit as pk

s = np.load(sys.argv[1]); A, B, dt = s['A'], s['B'], float(s['dt'])
m = bs.stage(sys.argv[2], sys.argv[3]); d = mujoco.MjData(m)
sl = bs.qpos_slices(m)
import roll_search as rs
order = list(rs.HOME)                       # joint order of the recorded qpos rows (Train asset)
src = mujoco.MjModel.from_xml_path(str(rs.ROBOT / 'jumper/assets/jumper/jumper.xml'))
src_adr = {n: src.jnt_qposadr[mujoco.mj_name2id(src, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in order}


def put(prefix, q):
    free, joints = sl[prefix]
    d.qpos[free:free + 7] = q[:7]
    for n, a in joints.items():
        d.qpos[a] = q[src_adr[n]]


w = imageio.get_writer(sys.argv[4], fps=30, quality=8, macro_block_size=8)
look = None
with mujoco.Renderer(m, 1080, 1920) as r:
    for f in range(int(len(A) * dt * 30)):
        i = min(int(round(f / 30 / dt)), len(A) - 1)
        put('A_', A[i]); put('B_', B[i]); mujoco.mj_forward(m, d)
        mid = 0.5 * (A[i, :3] + B[i, :3]); span = np.linalg.norm(A[i, :2] - B[i, :2])
        look = mid if look is None else look + 0.04 * (mid - look)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = (look[0], look[1], 0.12)
        cam.distance, cam.elevation, cam.azimuth = max(1.25, 0.75 + 0.75 * span), -24, 90
        opt = mujoco.MjvOption(); opt.geomgroup[1] = opt.geomgroup[3] = 0
        r.update_scene(d, cam, scene_option=opt)
        r.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1
        r.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 1
        w.append_data(r.render())
w.close()
