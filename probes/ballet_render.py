"""Render the Ballerina for X: a window of the trained jumper.dance_ballerina replay on the battle
stage (fog floor, soft light), a slow camera sweep round the dancer, fade in and out.

usage: ballet_render.py <replay.npz> <skin.xml> <t_start> <length> <out.mp4>
"""
import sys

import mujoco
import numpy as np
import imageio

import battle_stage as bs
import roll_search as rs

z = np.load(sys.argv[1]); Q, dt = z['qpos'], float(z['dt'])
t0, length = float(sys.argv[3]), float(sys.argv[4])
m = bs.stage(sys.argv[2]); d = mujoco.MjData(m)
free, joints = bs.qpos_slices(m)['A_']
src = mujoco.MjModel.from_xml_path(str(rs.ROBOT / 'jumper/assets/jumper/jumper.xml'))
adr = {n: src.jnt_qposadr[mujoco.mj_name2id(src, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in rs.HOME}
FPS, FADE = 30, 1.0
w = imageio.get_writer(sys.argv[5], fps=FPS, quality=8, macro_block_size=8)
look = None
with mujoco.Renderer(m, 1080, 1920) as r:
    for f in range(int(length * FPS)):
        tv = f / FPS
        q = Q[min(int(round((t0 + tv) / dt)), len(Q) - 1)]
        d.qpos[free:free + 7] = q[:7]
        for n, a in joints.items():
            d.qpos[a] = q[adr[n]]
        mujoco.mj_forward(m, d)
        look = q[:3].copy() if look is None else look + 0.05 * (q[:3] - look)
        u = tv / length
        cam = mujoco.MjvCamera()
        cam.lookat[:] = (look[0], look[1], 0.11)
        # a slow half-orbit from the front-left round to the front-right, easing in and out
        cam.azimuth = 135 + 90 * (0.5 - 0.5 * np.cos(np.pi * u))
        cam.elevation = -14 - 6 * np.sin(np.pi * u)
        cam.distance = 0.80 - 0.10 * np.sin(np.pi * u)
        opt = mujoco.MjvOption(); opt.geomgroup[1] = opt.geomgroup[3] = 0
        r.update_scene(d, cam, scene_option=opt)
        r.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1
        r.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 1
        img = r.render().astype(np.float32)
        g = min(1.0, tv / FADE, (length - tv) / FADE)
        w.append_data((img * max(0.0, g)).astype(np.uint8))
w.close()
