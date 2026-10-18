"""jumper.battle_gestures -- the calm half of a b-boy battle, learned by imitation.

The same environment as every `jumper.dance*` task. The clip, `media/battle_gestures.npz`
(33.6 s at 100 Hz, 108 BPM), is keyframed like jumper.dance_ballerina's and kept one consistent
posture by the same generator (support feet planted by IK, the CoM pulled over the support):
bobbing to the beat, the claws raised overhead and snapping like applause, the eyes covered,
a pinch toward the opponent (left claw snapping, right claw down as a support), a 180-degree
turn away and back on the walking tripods, and an uprock -- stepping in place.

Why a policy: open-loop PD tipped 18-40 degrees nose-down the moment both claws left the floor
(the CoM sits 1.7 cm in front of the middle feet); jumper.dance_ballerina's policy held arms-up
poses without that.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.battle_gestures",
    assets=JUMPER_ASSETS,
    description="jumper hexapod imitating b-boy battle gestures (bob, overhead claw applause, "
                "covered eyes, pinch, turn away and back, uprock); clip in media/",
    tags=("imitation", "jumper", "dance", "flat", "battle"),
)
