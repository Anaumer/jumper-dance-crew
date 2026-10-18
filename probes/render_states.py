"""Render recorded qpos states on a skin with a following camera:
render_states.py <states.npz> <skin.xml> <out.mp4> [azimuth] [elevation]"""
import sys
from pathlib import Path
import imageio
import mujoco
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'choreo'))
import posekit as pk

s = np.load(sys.argv[1]); Q, dt = s['qpos'], float(s['dt'])
m = pk.load(sys.argv[2]); d = mujoco.MjData(m)
az = float(sys.argv[4]) if len(sys.argv) > 4 else 170
el = float(sys.argv[5]) if len(sys.argv) > 5 else -10
step = max(1, int(round((1 / 30) / dt)))
w = imageio.get_writer(sys.argv[3], fps=1 / (step * dt), quality=8, macro_block_size=8)   # real time
with mujoco.Renderer(m, 720, 1280) as r:
    for i in range(0, len(Q), step):
        d.qpos[:] = Q[i]; mujoco.mj_forward(m, d)
        w.append_data(pk.render(m, d, r, azimuth=az, elevation=el, distance=0.9,
                                lookat=(float(d.qpos[0]), float(d.qpos[1]), 0.12)))
w.close()
