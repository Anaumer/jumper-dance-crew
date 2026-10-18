"""Where do the two crabs touch each other in a composed battle? (Each crab was simulated alone,
so any contact between them is unmodelled: it must be a light touch, like the fist bump, or none.)
usage: battle_contacts.py <battle.npz> <skin_a.xml> <skin_b.xml>"""
import sys
import mujoco
import numpy as np
import battle_stage as bs
import roll_search as rs
z = np.load(sys.argv[1]); A, B, dt = z['A'], z['B'], float(z['dt'])
m = bs.stage(sys.argv[2], sys.argv[3]); d = mujoco.MjData(m); sl = bs.qpos_slices(m)
src = mujoco.MjModel.from_xml_path(str(rs.ROBOT / 'jumper/assets/jumper/jumper.xml'))
adr = {n: src.jnt_qposadr[mujoco.mj_name2id(src, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in rs.HOME}
root = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)[:2]
hits = []
for i in range(0, len(A), 5):
    for p, Q in (('A_', A), ('B_', B)):
        free, joints = sl[p]; d.qpos[free:free + 7] = Q[i, :7]
        for n, a in joints.items():
            d.qpos[a] = Q[i, adr[n]]
    mujoco.mj_forward(m, d)
    worst = 0.0; pair = None
    for c in d.contact[:d.ncon]:
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if b1 and b2 and {root(b1), root(b2)} == {'A_', 'B_'}:
            if c.dist < worst:
                worst = c.dist; pair = (m.geom(c.geom1).name, m.geom(c.geom2).name)
    if pair:
        hits.append((i * dt, worst, pair))
print(f'{len(hits)} sampled frames with contact between the crabs')
for t, w, p in hits:
    print(f'  {t:6.2f} s  penetration {-w * 1000:5.1f} mm  {p}')
