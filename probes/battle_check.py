"""Per-second report of a composed battle: distance, and how far each crab looks away from the other."""
import sys
import numpy as np
import battle_stage as bs
z = np.load(sys.argv[1]); A, B, dt = z['A'], z['B'], float(z['dt'])
def off(P, O):
    y = np.array([bs.yaw_of(q[3:7]) for q in P]); to = O[:, :2] - P[:, :2]
    return np.degrees(np.abs((np.arctan2(to[:, 1], to[:, 0]) - y + np.pi) % (2 * np.pi) - np.pi))
oa, ob = off(A, B), off(B, A); dd = np.linalg.norm(A[:, :2] - B[:, :2], axis=1)
upA = np.array([1 - 2 * (q[4] ** 2 + q[5] ** 2) for q in A]); upB = np.array([1 - 2 * (q[4] ** 2 + q[5] ** 2) for q in B])
step = int(1 / dt)
for i in range(0, len(A), step):
    s = slice(i, i + step)
    print(f'{i*dt:5.0f}s  dist {dd[s].min():.2f}-{dd[s].max():.2f}  A looks off {oa[s].max():5.0f}°  B looks off {ob[s].max():5.0f}°  upA {upA[s].min():+.2f} upB {upB[s].min():+.2f}')
