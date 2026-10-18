"""jumper.freeze -- a one-arm breakdance freeze on the opened left claw, learned by imitation.

The same environment as every `jumper.dance*` task; this directory adds the clip and the numbers.

## The clip

`media/freeze.npz`, 11.45 s at 100 Hz (2.5 s standing at each end). The entry and the landing are **simulated, not keyframed**:
an open-loop joint-target schedule found by an evolution strategy in plain MuJoCo with this
robot's servo curve (`Robot/probes/move_search.py`, mode `freeze`) pushes the robot up onto its
opened left claw and lets it come back down onto all fours. Between the two, at the moment the
simulated move was closest to it, the clip blends (0.25 s) into a **statically stable** freeze
pose and holds it for 1.2 s: the opened claw's two prongs on the floor (13-18 cm2 of support),
the CoM inside that area, every arm joint at <= 48% of the torque plateau -- inside the 1.2 N*m
continuous rating, so the pose can be held. Under plain PD the open-loop move stays on the claw
alone for 0.7 s; holding longer is what the policy has to learn.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.freeze",
    assets=JUMPER_ASSETS,
    description="jumper hexapod imitating a one-arm freeze on the opened claw (simulated entry "
                "and landing, 1.2 s statically stable hold); clip in tasks/jumper/freeze/media/",
    tags=("imitation", "jumper", "dance", "flat", "freeze"),
)
