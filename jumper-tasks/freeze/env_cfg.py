"""Environment config for jumper.freeze: `media/freeze.npz`.

The environment is `tasks/jumper/common/dance/env.py`; this file holds every number.
Where a number follows a rule `jumper.dance` wrote down for its own clip, the rule is
applied to this clip's measurements and the result is quoted.

## This clip, measured

At the 50 Hz control rate (11.45 s, 2.5 s standing at each end): RMS joint amplitude 0.398 rad, joint velocity RMS 2.012 rad/s,
peaking at 27.2 rad/s (the push-up onto the claw); the base between 74 and 295 mm.

**The reward weights are starting points**: no policy has been trained on this clip yet.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.utils.spec_config import CollisionCfg

from ..common.constants import _LEG_BIT
from ..common.dance.env import dance_env_cfg

#: This task's material, written by `tools/import_wbc_dances.py`.
MEDIA = Path(__file__).resolve().parent / "media"

#: Episode length, seconds: nearly the whole clip. A freeze is one movement, and the adaptive
#: sampler still starts episodes inside it where it fails (reference-state init).
EPISODE_S = 8.0


def env_cfg(asset: Path | None = None, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Build this task's environment config. See `jumper.dance`'s for the arguments."""
    cfg = dance_env_cfg(
        media=MEDIA,
        asset=asset,
        play=play,
        episode_s=EPISODE_S,
        # ── Reference-state initialisation ────────────────────────────────
        # jumper.dance's: fractions of this robot (10 mm is a tenth of its standing
        # height), not of the clip, so they carry over unchanged.
        rsi_pose_range={
            "x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.005, 0.005),
            "roll": (-0.05, 0.05), "pitch": (-0.05, 0.05), "yaw": (-0.05, 0.05),
        },
        rsi_velocity_range={
            "x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.05, 0.05),
            "roll": (-0.2, 0.2), "pitch": (-0.2, 0.2), "yaw": (-0.2, 0.2),
        },
        rsi_joint_range=(-0.05, 0.05),
        # ── Observation noise ─────────────────────────────────────────────
        # The tracking skeleton's, as jumper.dance has them; not measured.
        joint_pos_noise=0.01,
        joint_vel_noise=0.5,
        # ── Rewards ───────────────────────────────────────────────────────
        # Positive budget 6.5 a step, as jumper.dance: mjlab's six tracking terms
        # unchanged (5.0) plus the two joint-space terms below (1.5).
        #
        # Joint position: jumper.dance's rule is a std at 0.44 of the clip's RMS
        # amplitude. Here: 0.44 x 0.398 -> 0.18.
        joint_pos_std=0.18,
        joint_pos_weight=1.0,
        # Joint velocity: two thirds of the clip's velocity RMS. Here: 2.012 -> 1.34.
        # Half the position weight, so when they disagree the pose wins.
        joint_vel_std=1.34,
        joint_vel_weight=0.5,
        # Smoothness: -0.1 is calibrated at this action scale (0.25) and the
        # second difference at half of it -- properties of the robot and the
        # action space, not of the clip.
        action_rate_weight=-0.1,
        action_acc_weight=-0.05,
        # Power: kept at the dances' -0.02. At this clip's ~0.3 rad/s it costs a few
        # hundredths a step -- the term only bites if the policy jitters.
        power_weight=-0.02,
        # Torque above the continuous rating: anchored to the servo, not the clip
        # -- one joint at the plateau costs 0.6, 9% of the budget.
        torque_headroom_weight=-2.0,
        # Joint limits: the reference stays inside every range by construction (the IK
        # clips to the limits); -10 as the dances, the soft limit keeps it off its stops.
        joint_limit_weight=-10.0,
        # Leg-on-leg contact above 1.0 N, ~8% of the robot's worst-case push.
        self_collision_weight=-1.0,
        self_collision_force=1.0,
        # ── Terminations ──────────────────────────────────────────────────
        # Root height: the base goes up to 295 mm in the freeze, so no threshold can exceed
        # the band and still catch a fall. 60 mm: a robot that falls out of the freeze is
        # well over that below the reference and is stopped there.
        anchor_height_error=0.06,
        # Relative to the reference's own orientation (mjlab compares projected gravity), so
        # being upside down in the freeze does not trip it. 0.5 is 60 degrees off the clip.
        anchor_tilt_error=0.5,
        # 80 mm, looser than the dances' 50: in the freeze the feet are in the air with the
        # body, and a few centimetres of lag there is tracking error, not a fall.
        support_foot_error=0.08,
        # ── Disturbance ───────────────────────────────────────────────────
        # jumper.dance's: gentler and rarer than mjlab's, sized to a 2 kg robot.
        push_interval_s=(3.0, 8.0),
        push_velocity_range={
            "x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.05, 0.05),
            "roll": (-0.2, 0.2), "pitch": (-0.2, 0.2), "yaw": (-0.2, 0.2),
        },
    )

    # ── Action scale ──────────────────────────────────────────────────────
    # 1.0 rather than the family's 0.25 (`constants.ACTION_SCALE`). The simulated entry
    # drives joints to targets up to ~3 rad from HOME (the push onto the claw is bang-bang,
    # found by search with targets anywhere in the joint ranges); at 0.25 that is an action
    # of ~12, which a Gaussian policy starting at std 1 reaches only very slowly -- the first
    # run (warm-started from jumper.dance_ballerina) sat at a mean episode of 7-10 steps for
    # 130 iterations, terminated on the feet at the push.
    for joint in cfg.actions["joint_pos"].scale:
        cfg.actions["joint_pos"].scale[joint] = 1.0

    # ── The claws collide with the crab's own torso ──────────────────────────────────
    # As jumper.battle_gestures: the family's masks collide a leg only with the ground and
    # the other legs. The first freeze policy (freeze_v2/model_2999) pressed the standing
    # claw's finger up to ~10 mm into the shell and display from below; the clip itself
    # (a physics take with self-collision on) overlaps the hulls by 2.6 mm at most. The
    # shell, display, ToF and camera hulls accept both arm bits; base_link stays out.
    robot = cfg.scene.entities["robot"]
    robot.collisions = tuple(_arm_torso_collision(c) for c in robot.collisions)
    return cfg


#: The torso hulls an arm can reach.
_TORSO_RE = "^(upper_shell|display_module|tof_sensor|camera)_link_meshcol$"


def _arm_torso_collision(base: CollisionCfg) -> CollisionCfg:
    arm_bits = _LEG_BIT["LF"] | _LEG_BIT["RF"]

    def with_torso(d: dict[str, int], value: int) -> dict[str, int]:
        # Patterns match first-hit in insertion order: the torso entry goes before ".*".
        out = {k: v for k, v in d.items() if k != ".*"}
        out[_TORSO_RE] = value
        out[".*"] = d[".*"]
        return out

    return replace(
        base,
        contype=with_torso(base.contype, base.contype[".*"]),
        conaffinity=with_torso(base.conaffinity, arm_bits),
    )
