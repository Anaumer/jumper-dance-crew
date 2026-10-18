"""Ballerina choreography v3 for Jumper (~37 s). v1 plus a stepping pirouette and a penche.

1  Preparation      bras bas, stillness
2  Port de bras     en avant -> a la seconde -> en haut
3  Plie / releve
4  Epaulement       sway left and right, arms answering
5  Pirouette        claws down, a full 360-degree tripod turn on the spot, body raised
6  Arabesque        CoM over the middle feet, rear legs back: two legs
7  Arabesque penchee  the right claw comes down in front as a third point, body leans in,
                    left claw high -- a claw-supported figure
8  Reverence
"""
import numpy as np
from posekit import HOME
from generate2 import Turn

BAS = [HOME[f'LF_J{k}_joint'] for k in range(5)]
AVANT = [-0.55, -1.8, -0.44, -1.37, 0.0]
AVANT_HAUT = [-0.55, -2.4, -0.44, -1.37, -0.3]
SECONDE = [-0.55, -3.0, -0.44, 0.5, -0.3]
HAUT = [-0.55, -3.0, -0.44, -1.37, -0.5]
HAUT_SOFT = [-0.55, -3.15, -0.30, -1.20, -0.7]
REVERENCE = [-0.80, -1.6, -0.9, -1.0, -0.3]
REAR_ARABESQUE = [1.2, 1.6, -1.6]
REAR_HOME = [HOME['LR_J0_joint'], HOME['LR_J1_joint'], HOME['LR_J2_joint']]
Z = [0, 0, 0, 0, 0, 0]
MID = [0, 0, 1, 1, 0, 0]            # support weights: middle feet only
PENCHE = [0, 1, 1, 1, 0, 0]         # middle feet + right claw

TURN_T0, TURN_CYCLES, TURN_HALF = 16.2, 10, 0.36
TURNS = [Turn(TURN_T0, 2 * np.pi, TURN_CYCLES, TURN_HALF, lift=0.035)]
TURN_T1 = TURNS[0].t1                                    # 23.4

KEYS = [
    dict(t=0.0, base=Z, larm=BAS, rarm=BAS, rear=REAR_HOME, com=0.0, lift=0.0,
         lf_plant=0.0, rf_plant=0.0, support=MID),
    dict(t=1.5, base=Z, larm=BAS, rarm=BAS),
    dict(t=3.5, base=[0, 0, 0.004, 0, 0, 0], larm=AVANT, rarm=AVANT),
    dict(t=5.5, base=[0, 0, 0.008, 0, 0, 0], larm=SECONDE, rarm=SECONDE),
    dict(t=7.5, base=[0, 0, 0.010, 0, 0, 0], larm=HAUT, rarm=HAUT),
    dict(t=9.0, base=[0, 0, -0.012, 0, 0, 0], larm=SECONDE, rarm=SECONDE),        # plie
    dict(t=10.8, base=[0, 0, 0.022, 0, 0, 0], larm=HAUT_SOFT, rarm=HAUT_SOFT),    # releve
    dict(t=12.4, base=[0, 0, 0.015, 0.07, 0, 0.14], larm=HAUT, rarm=SECONDE),      # epaulement
    dict(t=14.0, base=[0, 0, 0.015, -0.07, 0, -0.14], larm=SECONDE, rarm=HAUT),
    # claws come down onto the floor for the turn
    dict(t=15.4, base=[0, 0, 0.008, 0, 0, 0], larm=AVANT, rarm=AVANT, lf_plant=0.0, rf_plant=0.0),
    dict(t=16.1, larm=BAS, rarm=BAS, lf_plant=1.0, rf_plant=1.0),
    dict(t=TURN_T0 + 0.6, base=[0, 0, 0.012, 0, 0, 0]),                          # rise into it
    dict(t=TURN_T1 - 0.6, base=[0, 0, 0.012, 0, 0, 0]),
    dict(t=TURN_T1 + 0.2, base=[0, 0, 0.010, 0, 0, 0], lf_plant=1.0, rf_plant=1.0),
    # arms up again, weight onto the middle feet, arabesque
    dict(t=TURN_T1 + 1.2, larm=HAUT, rarm=HAUT, lf_plant=0.0, rf_plant=0.0, com=0.0,
         rear=REAR_HOME, lift=0.0),
    dict(t=TURN_T1 + 2.4, com=1.0, support=MID),
    dict(t=TURN_T1 + 3.8, rear=REAR_ARABESQUE, lift=1.0, larm=HAUT_SOFT, rarm=AVANT_HAUT),
    dict(t=TURN_T1 + 5.0, larm=HAUT, rarm=AVANT_HAUT),
    dict(t=TURN_T1 + 6.0, larm=HAUT_SOFT, rarm=AVANT_HAUT, rf_plant=0.0, support=MID,
         base=[0, 0, 0.010, 0, 0, 0]),
    # penche: right claw to the floor in front, body leans in, left claw high
    dict(t=TURN_T1 + 7.4, rf_plant=1.0, support=PENCHE, larm=HAUT,
         base=[0, 0, 0.002, 0, 0.10, 0]),
    dict(t=TURN_T1 + 9.0, larm=HAUT_SOFT, base=[0, 0, 0.000, 0, 0.12, 0]),
    dict(t=TURN_T1 + 10.4, rf_plant=0.0, support=MID, larm=HAUT, rarm=AVANT_HAUT,
         base=[0, 0, 0.010, 0, 0, 0]),
    dict(t=TURN_T1 + 11.6, rear=REAR_HOME, lift=0.0, larm=HAUT, rarm=HAUT),
    dict(t=TURN_T1 + 12.6, com=0.0),
    dict(t=TURN_T1 + 14.2, base=[0, 0, -0.008, 0, 0.12, 0], larm=REVERENCE, rarm=REVERENCE),
    dict(t=TURN_T1 + 15.8, base=Z, larm=BAS, rarm=BAS),
    dict(t=TURN_T1 + 16.8, base=Z, larm=BAS, rarm=BAS),
]
DURATION = TURN_T1 + 16.8
