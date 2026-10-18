"""Drive the shipped locomotion policy (out/bundle_example/jumper/models/locomotion.onnx, trained
on jumper.posture) in its own env, with commands from a Python function instead of the pad.

The env builds the 415-float actor observation exactly as in training; the ONNX carries the
normalisation. Commands go in through the operator hook the env already has: each command term
asks its operator for values every step, and a stub operator answers with ours.

    run(cmd, T) -> dict(t, qpos [T, nq] in jumper.xml order, dt)
    cmd(t, pos_xy, yaw) -> (vx, vy, wz, twist, pitch, roll, height)   body frame, m/s, rad/s, m
"""
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

ROBOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROBOT / 'jumper'))
from mjrl.backend.resolve import resolve  # noqa: E402
from mjrl.backend.select import use_backend  # noqa: E402

use_backend(resolve(backend='native', device='cpu', num_envs=1))
import onnxruntime as ort  # noqa: E402
import tasks  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402

ONNX = ROBOT / 'jumper/out/bundle_example/jumper/models/locomotion.onnx'
STAND_H = 0.10647


class _Stub:
    def __init__(self):
        self.values = {'twist': [0.0, 0.0, 0.0], 'posture': [0.0, 0.0, 0.0, STAND_H]}

    def command(self, term, stamp=None):
        return list(self.values[term])


_env = None


def env():
    global _env
    if _env is None:
        cfg = tasks.load_env_cfg('jumper.posture', asset=None, play=True)
        cfg.scene.num_envs = 1
        cfg.episode_length_s = 10_000.0
        e = ManagerBasedRlEnv(cfg=cfg, device='cpu')
        stub = _Stub()
        for n in ('twist', 'posture'):
            e.command_manager.get_term(n)._operator = stub
        sess = ort.InferenceSession(str(ONNX))
        m = mujoco.MjModel.from_xml_path(str(ROBOT / 'jumper/assets/jumper/jumper.xml'))
        names = list(e.scene['robot'].joint_names)
        qadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names]
        _env = (e, stub, sess, m, qadr)
    return _env


def yaw_of_quat(q):
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def run(cmd, T):
    e, stub, sess, m, qadr = env()
    obs, _ = e.reset()
    robot = e.scene['robot']
    inp = sess.get_inputs()[0].name
    n = int(round(T / e.step_dt))
    Q = np.zeros((n, m.nq))
    for s in range(n):
        p = robot.data.root_link_pos_w[0].numpy(); q = robot.data.root_link_quat_w[0].numpy()
        c = cmd(s * e.step_dt, p[:2].copy(), yaw_of_quat(q))
        stub.values['twist'] = list(c[:3]); stub.values['posture'] = list(c[3:])
        a = sess.run(None, {inp: obs['actor'].numpy().astype(np.float32)})[0]
        Q[s, :3] = p                                   # the state the command was computed on
        Q[s, 3:7] = q
        Q[s, qadr] = robot.data.joint_pos[0].numpy()
        obs, _, term, trunc, _ = e.step(torch.from_numpy(a))
        if bool(term.any()) or bool(trunc.any()):
            raise RuntimeError(f'episode ended at step {s}')
    return dict(t=np.arange(n) * e.step_dt, qpos=Q, dt=e.step_dt)


if __name__ == '__main__':
    # self-test: turn in place, then walk sideways round a circle facing its centre, bobbing
    def cmd(t, xy, yaw):
        bob = STAND_H + 0.012 * np.sin(2 * np.pi * t / (60 / 108))
        if t < 3:
            return (0, 0, 2.0, 0, 0, 0, bob)
        return (0, 0.2, 0.2 / 0.45, 0, 0, 0, bob)
    r = run(cmd, 9.0)
    Q = r['qpos']
    yaw = np.unwrap([yaw_of_quat(q) for q in Q[:, 3:7]])
    for t in (0.5, 1, 2, 3, 5, 7, 9 - r['dt']):
        i = int(t / r['dt']) - 1
        print(f't {t:4.1f}  xy {Q[i,0]:+.3f} {Q[i,1]:+.3f}  z {Q[i,2]:.3f}  yaw {np.degrees(yaw[i]):7.1f}')
    np.savez(Path(__file__).parent / 'loco_selftest.npz', **r)
