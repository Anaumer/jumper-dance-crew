# Jumper Dance Crew 🦀

A b-boy battle (and a ballet) for the **Tiao Tiao Jumper** crab robot
([KingKongRobotics/jumper](https://github.com/KingKongRobotics/jumper), 22 DoF), made for the
*Crab Robot Open-source Challenge 2026*. Everything you see in the video is **physics simulation
in MuJoCo** with the open-source Jumper model and the real servo's torque/speed curve, not
animation.

| | Video | What it shows |
|---|---|---|
| **B-Boy Battle** | [YouTube](https://www.youtube.com/watch?v=1j4DzeQWZX4) | two crabs, uprocks, claw fist bump, barrel roll, cartwheel, one-claw freeze, backspin on the shell, half-circle "courtship" dancing round the rival |

## What is in here

```
jumper-tasks/     three imitation tasks for the jumper framework + install.py
  battle_gestures/  bob, overhead applause, covering the eyes, pinch, uprock
  freeze/           one-claw freeze: simulated entry, held pose, landing
  dance_ballerina/  ballet with a two-leg arabesque and a pirouette
policies/         the trained PPO checkpoints (rsl_rl, warm-started from one another)
choreo/           keyframe -> clip generators (IK-planted feet, CoM pull, tripod turns) and
                  the battle and ballet choreographies
probes/           physics search and composition:
  roll_search.py, move_search.py   evolutionary search of open-loop moves (barrel roll, cartwheel,
                                   freeze entry) against the measured servo curve
  headspin_search.py, headspin_chain.py   the backspin: lie down, 3+ turns pushed by the claws,
                                   flip back up -- one continuous simulation
  loco.py          drives the locomotion policy that ships with jumper (ONNX) from Python
  battle3.py       the battle: two timelines, closed-loop gaze, half-circle orbits, music grid
  battle_render.py, ballet_render.py   video rendering
  clean_poses.py, self_pen.py, battle_contacts.py, battle_check.py   checks (see below)
skins/tools/      paint-only skins (camo, black stripes, ballerina, Swiss, China ...) for the
                  jumper-design pipeline
```

## How it works

**Moves are physics, not keyframes.** Three kinds of motion:

1. **Imitation policies** (PPO, the framework's BeyondMimic-style dance tracking): a choreography
   is written as keyframes (`choreo/choreography_*.py`), turned into a kinematically consistent
   clip with IK-planted feet (`choreo/generate*.py`), and a policy learns to perform it in
   simulation with pushes and noise. Battle gestures, freeze and ballet are such policies.
2. **Open-loop physics moves** found by an evolution strategy over keyframe schedules
   (`probes/move_search.py`, `headspin_search.py`): barrel roll, cartwheel, backspin. Whatever
   wins is by construction something the simulated robot can do. The key idea for the rolls was
   to rock to one side first to build momentum.
3. **The shipped locomotion policy** (`locomotion.onnx` from the jumper bundle) for walking,
   orbiting and turning, steered closed-loop in `probes/loco.py` / `battle3.py` so the waiting
   crab always faces the dancer.

The battle is assembled from these segments (`probes/battle3.py`): each segment is placed
rigidly where the last one ended and cross-faded over 0.1 s; every power move starts on a beat
of the track (measured: 90.0 BPM, drop at 10.04 s).

**Checks that run on every result**

- `clean_poses.py` / `self_pen.py`: no claw inside the crab's own shell, display, ToF or camera.
  The framework's collision masks only collide a leg with the ground and the *other* legs, so
  the first policies tracked an overhead clap 30 mm into the shell. The tasks here therefore add
  arm-vs-torso contact (see `_arm_torso_collision` in each `env_cfg.py`); the final replays have
  0 frames with a claw in the body.
- `battle_contacts.py`: the two crabs are simulated separately, so they must not touch -- except
  the fist bump, placed mirror-symmetrically so the claws just meet.
- `battle_check.py`: per second, how far each crab looks away from the other.

**Measured results** (replays of the checkpoints in `policies/`)

| | |
|---|---|
| freeze | 1.44 s standing on the opened left claw alone (1.60 s with small hops), lands on all fours |
| battle gestures | yaw drift below 1.5 degrees over pinch, applause and covering the eyes |
| backspin | ~3.8 turns in about 2 s on the shell at the model's own friction, flip back to standing |
| ballet | two-leg arabesque held 2.2 s, a 337-degree pirouette |

## Run it

```bash
git clone https://github.com/KingKongRobotics/jumper.git
git clone <this repo> jumper-dance-crew && cd jumper-dance-crew
ln -s ../jumper jumper          # the scripts expect jumper/ (and jumper-design/ for skins) beside them
python jumper-tasks/install.py jumper
cd jumper && pip install -e ".[dev]" && cd ..      # see jumper's docs/AGENT_SETUP.md

# replay a policy (writes the qpos sequence and per-limb clearances)
cd probes
python eval_task.py ../policies/battle_gestures_model_14047.pt gestures_replay.npz 1679 jumper.battle_gestures

# train further (GPU: --backend warp; on a Mac: --backend native --device cpu)
cd ../jumper && python scripts/train.py --task jumper.battle_gestures --num_envs 4096 \
    --resume --checkpoint ../policies/battle_gestures_model_14047.pt --headless

# compose and render the battle (needs the two skins built with skins/tools + jumper-design)
cd ../probes
python battle3.py battle_replay_v2.npz freeze_v3_replay.npz battle.npz
python battle_render.py battle.npz <camo skin xml> <stripes skin xml> battle.mp4
```

Training ran on one rented NVIDIA A40 (about 1.6 s per PPO iteration with 4096 environments);
all policies together cost a few USD of GPU time. The search scripts run on a laptop CPU.

## Honest notes

- This is simulation only; nothing here has run on a real Jumper yet. The servo model, PD gains
  and limits are the framework's, so the motions respect the real actuator, but sim-to-real was
  not tested.
- Segments of the battle are stitched with short cross-fades; within a segment every frame is
  simulated.

## License and credits

Code: Apache License 2.0, like the jumper framework it builds on (see `LICENSE`). The task
configs started as copies of jumper's own dance tasks. Music is **not** included: the video uses
"Turn Back" by Ballpoint (Epidemic Sound).

Made with Claude Code. #Jumper #CrabRobot #OpenSourceRobot
