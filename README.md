# SO-101 Robot Arm: Reinforcement Learning in MuJoCo, Deployed to Hardware

[![tests](https://github.com/jwei11/robot-arm/actions/workflows/tests.yml/badge.svg)](https://github.com/jwei11/robot-arm/actions/workflows/tests.yml)
![Python 3.10](https://img.shields.io/badge/python-3.10-3776ab)
![MuJoCo](https://img.shields.io/badge/sim-MuJoCo%203.3-555)
![PPO](https://img.shields.io/badge/RL-PPO%20(Stable--Baselines3)-555)
![LeRobot](https://img.shields.io/badge/hardware-LeRobot-555)
[![License: MIT](https://img.shields.io/badge/license-MIT-555)](LICENSE)

A 5-joint [SO-101](https://github.com/TheRobotStudio/SO-ARM100) robot arm learns two skills
with reinforcement learning in simulation: moving its gripper to any target in front of it,
and pushing a block onto a goal. The reach policy also runs on the physical arm through
[LeRobot](https://github.com/huggingface/lerobot), behind several layers of safety checks.

<p align="center">
  <img src="docs/reach_demo.gif" alt="Simulated SO-101 arm reaching three random targets" width="400">
  <img src="docs/push_demo.gif" alt="Simulated SO-101 arm pushing a block onto a goal" width="400">
</p>

## At a glance

- **Built:** custom Gymnasium environments for reaching and block pushing on the official
  SO-101 MuJoCo model, PPO training and evaluation scripts, and a sim-to-real runner for the
  physical arm.
- **Reach:** 100% success (tip within 2 cm) on 200 held-out targets, median miss 0.12 cm;
  99.5% with randomized physics, sensor noise and control delay. Trains in about 17 minutes
  on a laptop CPU.
- **Push:** a reward redesign took success from 4.5–6% (two seeds) to 39–87.5% (three seeds,
  mean 70%) at the same training budget.
- **How:** each improvement came from analysing failures, not from training longer: fixing
  a reward that paid for touching the block instead of pushing it, and changing where
  training targets come from.
- **Evaluated carefully:** models compared on the same held-out test set, several seeds per
  setting, and "best checkpoint" scores re-checked on fresh episodes.
- **Real hardware:** a tested joint-to-servo mapping and safety layers (dry run by default,
  joint clamping, rate limiting, typed confirmation, safe stop). 56 automated tests run on
  every push.

**Stack:** Python, PyTorch, Stable-Baselines3 (PPO), Gymnasium, MuJoCo, LeRobot, pytest,
GitHub Actions.

<p align="center"><img src="docs/results.svg" alt="Bar charts. Pushing: first reward 4.5% and 6%, redesigned reward 39% and 84% for seeds 0 and 1. Reaching: 90% with targets all around, 90% front only, 100% front only and evenly spread." width="760"></p>

## Results

| Task | Result (simulation) |
|---|---|
| Reach a random 3D target | **100% success** (tip within 2 cm) on 200 held-out targets; median final distance **0.12 cm** |
| Same, with randomized physics, sensor noise and control delay | 99.5% success (trained without randomization) |
| Training cost | 2M steps, about 17 minutes on CPU (8 parallel environments) |
| Push a block onto a goal (within 3 cm) **84% and 87.5%** on 200 held-out episodes for two of three seeds (the third reached 39%), vs. 4.5% and 6% for the first reward design at the same 20M-step budget (see [Push task](#push-task)) |

The policy has also run on a physical SO-101 follower arm. Real-world accuracy hasn't been
measured yet.

### How success reached 100%

All three models below were scored on the **same** 200 held-out targets (front workspace,
spread evenly):

| Model | Training targets | Success | Median miss |
|---|---|---|---|
| Original | all around the arm, random joint angles | 90.0% | 0.46 cm |
| Scoped to the front | front only, random joint angles | 90.0% | 0.19 cm |
| **Even coverage** | front only, spread evenly over the reachable space | **100%** | **0.12 cm** |

Scoping to the front matched training to the real job but didn't by itself improve
accuracy. Even coverage of the workspace is what took success from 90% to 100%. Each row is
a single training run (one seed).

Earlier runs, each scored on its own training distribution:

| Change | Steps | Success |
|---|---|---|
| Original reward, default policy noise | 2.4M | 80% |
| Lower initial exploration noise + a precision reward term | 1.5M | 83% |
| Same with 16 environments instead of 8 | 3M | 81.5% (targets all around the arm) |

1. **Check the task is solvable first.** Early training learned slowly. A scripted inverse-kinematics
   controller using the same action interface reached 86%, which showed the problem was the
   learning setup, not the task. Starting with less random exploration (`log_std_init = -1`) and
   adding a reward term that rises sharply in the last few centimetres roughly halved the steps
   needed.
2. **More parallel environments isn't free.** With 16 environments each PPO iteration collects
   twice as much data, so over the same number of steps the policy is updated half as often.
   It learned more slowly per step (48% vs. 83% at 1.5M).
3. **Break failures down before training longer.** 27 of the 37 misses at 81.5% were targets
   behind the arm, which need it to flip over the top. The arm's real tasks are all in front,
   so the task was scoped to the front half.
4. **The training distribution is part of the model.** One remaining failure was reachable but
   lay where random *joint angles* almost never put a target (0 of 3,000 samples within 3 cm).
   `WorkspaceSampler` instead picks a 3 cm cell of the reachable space uniformly, then a point
   inside it. On the same test targets, that took success from 90% to 100%.

## Quick start (Windows, PowerShell)

Requires Python 3.10.

```powershell
python3.10 -m venv .venv
```
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
```powershell
.\.venv\Scripts\python.exe -m pytest
```

Train and evaluate the reach policy:
```powershell
.\.venv\Scripts\python.exe train.py --run-name reach --timesteps 2000000
```
```powershell
.\.venv\Scripts\python.exe evaluate.py --model runs\reach\best_model.zip --episodes 200
```
`evaluate.py` prints success rate and final distances and saves `runs\reach\rollouts.mp4`.
Training progress is in TensorBoard (`.\.venv\Scripts\tensorboard.exe --logdir runs`).
Every PPO and environment setting is a flag (`train.py --help`).

For the real arm, also install LeRobot with Feetech servo support:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-real.txt
```

`requirements.txt` pins direct dependencies; `requirements-lock.txt` is the full tested
environment. PyTorch is the CPU build on purpose: with a 256×256 network the bottleneck is
physics simulation, and moving tiny batches to a GPU is slower.

## Project layout

```
so101_rl/
  env.py          reach environment (Gymnasium); observation/action helpers shared with run_real.py
  push_env.py     push-a-block environment
  config.py       environment settings and domain randomization (off by default)
  kinematics.py   model loading, forward kinematics, damped-least-squares IK
  targets.py      reachable target sampling, including the even-coverage WorkspaceSampler
  mapping.py      MuJoCo radians <-> LeRobot joint values
  safety.py       joint clamping and per-step rate limiting for real commands
  tasks.py        task name -> config and environment class
train.py          PPO training (--task reach|push)
evaluate.py       success rate, distance statistics, rollout video
run_real.py       runs a trained reach policy on the physical arm (dry run by default)
assets/so101/     official SO-101 MuJoCo model and meshes, plus reach and push scenes
tests/            pytest suite (56 tests)
```

## Reach environment

| | |
|---|---|
| Control | 20 Hz (25 × 2 ms physics steps); 100-step (5 s) episodes |
| Action | 5 values in [-1, 1]: change in each joint's position target, at most 0.05 rad (2.9°) per step, clamped to joint limits minus 5°. The gripper is held closed. |
| Observation (24) | joint angles, joint velocities, tip position, target position, scaled tip-to-target vector, previous action |
| Reward | `-d + 0.5·[d < 2 cm] + (1 - tanh(d / 5 cm)) - 0.01·|a|²`, where d is the tip-to-target distance in metres |
| Targets | reachable by construction (forward kinematics of a valid, collision-free joint configuration), spread evenly over the reachable space, at least 5 cm above the table and within 90° of straight ahead |
| Tip | the model's `gripperframe` site, between the jaws |

Design choices:
- **Joint position changes, not torques.** The real servos are position-controlled, so the policy
  speaks the hardware's language, and small per-step changes are naturally smooth.
- **Observations come only from joint angles.** Tip position and velocity are computed from
  them, because the real arm reports positions only. Simulation and hardware share one
  `build_observation()` function.
- **Domain randomization hooks** (off by default; `--dr` turns on a preset): joint damping,
  armature and friction; servo strength (0.4–1.2×, which covers 7.4 V servos on a 5 V supply);
  joint-angle noise; and 0–2 steps of control delay.

## Push task

<p align="center"><img src="docs/push_demo.gif" alt="Simulated SO-101 arm pushing a blue block onto a green goal disc" width="432"></p>

`train.py --task push` trains the same arm, with the same actions, to push a 4 cm, 50 g cube
onto a goal disc 6–15 cm away. Episodes are 10 s, and success means the block ends within 3 cm
of the goal. This task is simulation only; `run_real.py` refuses push models.

| Reward design | Steps | Success (held-out episodes) | What the policy did |
|---|---|---|---|
| Reward for being near the block, plus distance to goal | 1.5M | 6% (100) | touched the block and shoved it in random directions |
| Reward for reaching a point *behind* the block (away from the goal), plus per-step progress | 1.5M | 8% (100) | pushed the block further, not yet toward the goal |
| Same, seed 0 | 20M / 30M (5.3 h) | 39% / 42.5% (200) | plateaued at 40–45% from 5M steps on |
| **Same, seed 1** | 20M (4.4 h) | **84%** (200) | median final block-to-goal distance 1.6 cm |
| **Same, seed 2** | 20M (4.3 h) | **87.5%** (200) | median final block-to-goal distance 1.0 cm |
| Same reward + goal-distance curriculum, seed 0 | 20M (4.5 h) | 77% (200) | still improving when training ended |
| Same + curriculum, seed 1 | 13M (crashed at 14.8M) | 4.5% (200) | never left the easiest stage; the same seed without curriculum scored 81.5% at 13M |
| First design again, seeds 0 and 1 (same-budget baseline) | 20M each (4.8 h) | 4.5% and 6% (200) | still touching and shoving after 20M steps |

The first design is a classic reward-shaping trap: touching the block earned reward on its
own, so the policy learned to touch rather than push. More training does not fix it: at the same
20M steps, the first design reached 4.5% and 6% on the two seeds, against 39% and 84% for
the second.

**Run-to-run variation is the biggest effect.** The same code and settings gave 39%, 84% and
87.5% on three random seeds, all at 20M steps, so one run per setting can't show whether a
change helped. The curriculum is the clearest example: on seed 0 it beat the same-seed
baseline (77% vs. 39%), but on seed 1 it never got training success on the easiest goals past
about 10%, stayed on that stage, and scored 4.5% against 81.5% for the same seed without it.
A single run would have suggested either "the curriculum doubles success" or "the curriculum
breaks training".

Other findings from the seed-0 run (42.5% after 30M steps):
- **Near misses and overshoot.** 52% of episodes got the block into the goal at some point,
  but 9.5% pushed it back out.
- **Longer pushes are harder:** 50% success when the block started 6–9 cm from the goal,
  31–38% beyond that.
- **Picking the "best" checkpoint misled.** The checkpoint that scored 60% on a 50-episode
  check scored 38% on 200 fresh episodes. Periodic checks now use 100 episodes, and final
  numbers are always on 200 fresh episodes.

`train.py --task push --curriculum` starts goals 4–7 cm from the block and widens the range in
four steps up to 6–15 cm, each time training success over the last 200 episodes reaches 50%.
Evaluation always uses the full range. Likely next steps: more seeds per setting, a curriculum
that moves on by time as well as by success rate, a reward term for holding the block in the goal, or SAC.

## Running on the real arm

`run_real.py` drives a calibrated SO-101 follower through LeRobot 0.4.4. **By default it is a
dry run:** it connects, reads the arm, and prints every command it *would* send.

1. Read-only mapping check. Renders what the simulator thinks the arm's pose is
   (`mapping_check.png`) so it can be compared with the real arm. Nothing moves.
   ```powershell
   .\.venv\Scripts\python.exe run_real.py --check-mapping
   ```
2. Dry run of the policy:
   ```powershell
   .\.venv\Scripts\python.exe run_real.py --model runs\reach\best_model.zip
   ```
3. Live, one target first (asks you to type `YES`):
   ```powershell
   .\.venv\Scripts\python.exe run_real.py --model runs\reach\best_model.zip --live --targets 0
   ```

The arm moves slowly to a home pose, runs the policy toward each of five fixed safe targets
(returning home in between), then returns to its starting pose. `--fake` runs the whole
sequence against a simulated arm.

**Safety layers, every control step:**
- policy action clipped to its valid range
- joint targets clamped to the simulator's limits *and* the arm's calibrated range minus 3°
  (LeRobot doesn't clamp in degrees mode)
- at most 1.5° change per step (30°/s), with a hard cap of 3°
- LeRobot's own `max_relative_target` (5° from the measured position)
- targets only from a fixed list
- **Ctrl+C** stops the policy and slowly returns to the start pose; a second Ctrl+C holds the
  current pose

### Joint units and mapping

LeRobot 0.4.4 reports joints in degrees by default:
`deg = (raw - mid) · 360 / 4095`, where `mid` is the middle of the calibrated range.
The official MuJoCo model also puts each joint's zero at mid-range, so the mapping is
`q_mujoco = deg2rad(lerobot_deg)` with no sign flips or offsets. The evidence:
- the model's zero convention matches LeRobot's
- LeRobot's own SO-101 kinematics uses plain `deg2rad` with the same model
- the calibrated joint ranges match the model's
- recorded real poses all land inside the model's limits, and flipping a joint's sign puts
  them in the floor
- `pose_check.py` renders were checked by eye against the real arm

The tests check the mapping against LeRobot's actual normalize/unnormalize code in both unit
modes, including rounding to integer servo ticks. `--sign joint=-1` overrides a joint if a
rebuilt arm turns the other way.

Known sim-to-real differences: the simulated servos are full strength; the real arm runs at
about half the policy's training speed because of the 1.5° safety cap; and gear backlash and
calibration error aren't modelled.

## Other hardware scripts

Utilities written while bringing up the arm. All use LeRobot on `COM20` with calibration id
`my_follower`; change these for your setup.

| Script | What it does |
|---|---|
| `pose_check.py` | read-only, torque off: renders the simulator's view of the current pose |
| `keyboard_drive.py` | drive one joint at a time from the keyboard |
| `xyz_control.py` | move the gripper in x/y/z centimetres using the model's kinematics |
| `teach_replay.py` | pose the arm by hand at waypoints, then replay them smoothly |
| `record_motion.py` | record a hand-guided motion at 50 Hz and play it back (`motions/`) |
| `monitor.py`, `arm_guard.py` | live servo telemetry and bump detection |
| `bump_test.py`, `nudge_test.py` | measure tracking error and load; small motion test |

## License

MIT for this project's code (see [LICENSE](LICENSE)). The SO-101 model files in
`assets/so101/` keep their own Apache-2.0 license.

## Acknowledgements

The SO-101 MuJoCo model and meshes in `assets/so101/` are from
[TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) (commit 141aea7),
Apache-2.0; see `assets/so101/LICENSE`. Training uses
[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) and
[Gymnasium](https://github.com/Farama-Foundation/Gymnasium); hardware access uses
[LeRobot](https://github.com/huggingface/lerobot).
