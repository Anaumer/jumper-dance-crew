"""Battle gestures for Jumper, one clip for a tracking policy (jumper.battle_gestures).

Order (at 108 BPM): stand, bob to the beat, clap the claws overhead on every beat, cover the
eyes, pinch toward the opponent (left claw snapping, right claw down as a support), turn away
180 degrees, bob, turn back, uprock (stepping in place on the tripods), stand.
The CoM is kept over the support (four legs, or legs plus claws) by the generator's `com`
channel -- open-loop PD tipped 18-40 degrees nose-down the moment both claws left the floor.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'probes'))
from posekit import HOME
from generate2 import Turn
import battle_moves as bm

BEAT = bm.BEAT
SIGN = bm.ARM_SIGN


def as_lists(pose):
    left = [pose.get(f'LF_J{k}_joint', HOME[f'LF_J{k}_joint']) for k in range(5)]
    right = [SIGN[k] * pose.get(f'RF_J{k}_joint', HOME[f'RF_J{k}_joint']) for k in range(5)]
    return left, right


BAS = [HOME[f'LF_J{k}_joint'] for k in range(5)]
# Overhead applause, from probes/clean_poses.py: the first pose (-0.55, -3.0, -0.44, -1.37) opened
# its pincers 18 mm into the shell (and the policy tracked it 30 mm in). This one swings the arms
# in over the head: open (J4 -1.0) and shut both stay >= 10 mm clear of shell, display, ToF and
# camera, the shut claws come within 5.5 mm of each other, and the tips sit 13 cm above the base.
HAUT_OPEN = [-1.813, -3.039, -1.103, -0.35, -1.0]   # claws up overhead, pincers open
HAUT_CLAP = [-1.813, -3.039, -1.103, -0.35, 0.0]    # ... snapped shut: the crab's applause
UP_SHUT = [-0.55, -3.0, -0.44, -1.37, 0.0]         # on the way up
EYES_L, EYES_R = as_lists(bm.P['eyes'])
JAB_L, _ = as_lists(bm.P['jab'])
JAB_L[4] = -1.0                         # -1.2 put the opening finger onto the ToF sensor (clean_poses.py)
JAB_SHUT = list(JAB_L); JAB_SHUT[4] = 0.0
Z = [0, 0, 0, 0, 0, 0]
FOUR = [0, 0, 6, 6, 1, 1]               # support weights: four legs, CoM kept ~2.5 cm behind the middle feet
                                        # (the four-leg centroid would pull the body 8 cm back, out of the legs' reach)
FOUR_RF = [0, 2, 6, 6, 1, 1]            # four legs and the right claw


def base(dz):
    return [0, 0, dz, 0, 0, 0]


# ── keys with explicit holds ─────────────────────────────────────────────────────────────
# generate.channel interpolates a channel between the keys that carry it, so a channel set at
# t=0 and next at 19 s ramps across all 19 s (measured: the claws sank and the CoM correction
# faded through the clap). `move` therefore writes every changed channel twice: its current value
# at the start of the transition, its new value at the end.
STATE = dict(base=Z, larm=BAS, rarm=BAS,
             rear=[HOME['LR_J0_joint'], HOME['LR_J1_joint'], HOME['LR_J2_joint']],
             com=0.0, lift=0.0, lf_plant=0.0, rf_plant=0.0, support=FOUR)
KEYS = [dict(t=0.0, **STATE)]


def move(t0, dur, **ch):
    KEYS.append(dict(t=t0, **{k: STATE[k] for k in ch}))
    STATE.update(ch)
    KEYS.append(dict(t=t0 + dur, **ch))
    return t0 + dur


def bob(t, beats, extra=None):
    """Dip on every beat, rise between; `extra(k)` adds per-half-beat channels (held explicitly)."""
    KEYS.append(dict(t=t, base=STATE['base'], **({k: STATE[k] for k in extra(0)} if extra else {})))
    for k in range(2 * beats):
        key = dict(t=t + (k + 1) * BEAT / 2, base=base(-0.012 if k % 2 == 0 else 0.004))
        if extra:
            key.update(extra(k)); STATE.update(extra(k))
        KEYS.append(key)
    STATE['base'] = KEYS[-1]['base']
    return t + beats * BEAT


t = 1.5
t = bob(t, 8)                                                            # bob, claws resting
# raise first (the old overhead pose, shut: 33 mm clear), then swing in over the head -- straight
# from BAS the two arms cross each other on the way up (45 mm, measured on the clip)
t = move(t, BEAT / 2, larm=UP_SHUT, rarm=UP_SHUT, com=1.0, base=base(0.0))
t = move(t, BEAT / 2, larm=HAUT_OPEN, rarm=HAUT_OPEN)
t = bob(t, 8, lambda k: dict(larm=HAUT_CLAP if k % 2 == 0 else HAUT_OPEN,
                              rarm=HAUT_CLAP if k % 2 == 0 else HAUT_OPEN))   # clap on the beat
t = move(t, BEAT, larm=EYES_L, rarm=EYES_R, base=base(0.0))
t = bob(t, 6)                                                            # cover the eyes
t = move(t, BEAT, larm=BAS, rarm=BAS, com=0.0, base=base(0.0))
t = move(t, BEAT, larm=JAB_L, rf_plant=1.0, support=FOUR_RF, com=1.0)
t = bob(t, 4, lambda k: dict(larm=JAB_L if k % 2 == 0 else JAB_SHUT))   # pinch, snapping
t = move(t, BEAT, larm=BAS, rf_plant=0.0, com=0.0, base=base(0.0), support=FOUR)
t = move(t, 0.4, lf_plant=1.0, rf_plant=1.0)                             # claws down as feet
t += 0.1
TURNS = [Turn(t, np.pi, 5, 0.30, lift=0.03)]
t = TURNS[0].t1 + 0.2
t = bob(t, 4)
TURNS.append(Turn(t, -np.pi, 5, 0.30, lift=0.03))                        # turn back
t = TURNS[1].t1 + 0.2
TURNS.append(Turn(t, 0.0, 6, BEAT / 2, lift=0.035))                      # uprock: step in place
t = TURNS[2].t1
t = move(t, 0.4, lf_plant=0.0, rf_plant=0.0, larm=BAS, rarm=BAS, base=base(0.0))
t = move(t, 2.0, base=Z)
DURATION = t
