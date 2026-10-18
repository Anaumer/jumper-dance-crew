"""jumper.dance_ballerina -- a ballet choreography with an arabesque on two legs, learned by imitation.

The same environment as every `jumper.dance*` task (`tasks/jumper/common/dance/env.py`);
this directory adds the clip and the numbers.

## The clip

`media/ballerina.npz`, 40.2 s at 100 Hz, generated from keyframes rather than recorded:
port de bras (claws en avant -> a la seconde -> en haut), plie and releve, epaulement,
a **360-degree pirouette** stepped on the robot's two walking tripods (claws down as feet),
an **arabesque on the two middle legs** (the CoM is pulled over the LM/RM line and the
rear legs extend backwards), an arabesque penchee with the right claw on the floor as a
third point, and a reverence. Every limb pose was checked against the Ballerina skin's
tutu envelope: nothing passes through it.

Every frame is one kinematically consistent posture: the support feet are held on the
HOME footprint by IK (worst residual 1 um), so `body_*` and `meas_*` are the same
channels -- there is no PD take behind them, and the support-foot check passes on
construction rather than by measurement. Quintic smoothstep between keyframes keeps
every joint under 6.1 rad/s.

The two-leg phase is the point of the task and the risk in it: a probe holding that
posture under plain PD falls in about 0.4 s (pitch-unstable on the LM/RM line), at
about 14% of the torque plateau. Balancing it is what the policy has to learn.

No music ships with it.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.dance_ballerina",
    assets=JUMPER_ASSETS,
    description="jumper hexapod imitating a keyframed ballet choreography with a two-leg "
                "arabesque; the clip is committed in tasks/jumper/dance_ballerina/media/",
    tags=("imitation", "jumper", "dance", "flat", "balance"),
)
