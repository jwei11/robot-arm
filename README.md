# SO-101 reach: MuJoCo + PPO, with a gated real-arm runner

Train the SO-101 follower arm in MuJoCo to put its gripper tip on a random 3D target,
evaluate it, and (carefully) run the trained policy on the real arm through LeRobot 0.4.4.

```
assets/so101/            official SO-101 MJCF + meshes (TheRobotStudio/SO-ARM100 @ 141aea7, Apache-2.0)
  so101_new_calib.xml    unmodified robot model
  reach_scene.xml        adds floor, lights, visual-only target marker
so101_rl/
  env.py                 SO101ReachEnv (Gymnasium), shared obs/action helpers
  config.py              EnvConfig + DomainRandomization (all off by default)
  kinematics.py          model loading, joint metadata, FK, IK (for checks)
  targets.py             reachable target sampling
  mapping.py             MuJoCo radians <-> LeRobot joint values
  safety.py              clamp + per-step rate limit for real commands
train.py  evaluate.py  run_real.py
tests/                   pytest suite (tests/data has a copy of the my_follower calibration)
```

Your earlier scripts (`keyboard_drive.py`, `nudge_test.py`, `teach_replay.py`) are untouched.

## Setup (Windows, PowerShell)

Python 3.10 from the Microsoft Store is fine; `py`/`python3.10` work even though its
Scripts folder is not on PATH, because everything runs through the venv's python.

```powershell
python3.10 -m venv .venv
```
```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
For the real arm (adds LeRobot 0.4.4 with Feetech support, CPU torchvision):
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-real.txt
```
`requirements.txt` pins the direct dependencies; `requirements-lock.txt` is the full
`pip freeze` of the tested sim environment. PyTorch is the **CPU** build on purpose: for
PPO with a 256x256 MLP the bottleneck is MuJoCo stepping, and CPU is faster than
shuttling tiny batches to the GPU.

## Tests and lint

```powershell
.\.venv\Scripts\python.exe -m pytest
```
```powershell
.\.venv\Scripts\python.exe -m ruff check .
```

## Train

```powershell
.\.venv\Scripts\python.exe train.py --run-name ppo_reach_ws --timesteps 2000000
```
(Defaults: 3M steps, 8 parallel envs, targets spread evenly over the reachable space within 90 deg of straight ahead, PPO lr 3e-4, rollout 512 x 16, batch 512, 10 epochs,
256x256 MLP, initial policy log-std -1.)
Watch it:
```powershell
.\.venv\Scripts\tensorboard.exe --logdir runs
```
Outputs in `runs\ppo_reach_ws\`: `best_model.zip` (best periodic eval), `final_model.zip`,
`checkpoints\`, `eval\evaluations.npz`, `env_config.json` (the exact env settings, read back
by `evaluate.py` and `run_real.py`). Add `--dr` to train with the sim-to-real randomization
preset. All PPO settings are flags (`--help`).

**How long, on this machine** (24 logical CPU threads, CPU-only torch): the default 3M-step
run took **20.9 min** (2,400 steps/s with 16 envs); 2M steps with 8 envs took **17.0 min** (1,960 steps/s).

**What to expect** (measured while building this):

| Run | Steps | Eval success (final distance < 2 cm) |
|---|---|---|
| current defaults | 1.0M | 72% |
| current defaults | 1.5M | 83% on 200 held-out episodes; median final distance 0.3 cm, mean 2.9 cm |
| original reward, default PPO noise (std 1) | 1.0M / 2.4M | 56% / 80% |
| all-around targets, 16 envs (`runs\ppo_reach`) | 3.0M | 81.5% on 200 held-out episodes (also 81.5% with `--dr`); median 0.4 cm |
| front targets, joint-space sampling, 8 envs (`runs\ppo_reach_front`) | 2.0M | **92.5%** on 200 held-out front targets; median 0.2 cm, p90 0.7 cm; ~94% from 650k on |
| **current defaults**: front targets spread evenly, 8 envs (`runs\ppo_reach_ws`) | 2.0M | **100%** on 200 held-out targets (99.5% with `--dr`); median 0.12 cm, p90 0.23 cm; 98% by 700k |

The precision term and the smaller initial policy noise roughly halved the steps needed.
The 1.5M rows used 8 envs; with 16 envs each PPO update sees twice the data, so there are half
as many updates per step and the curve was slower (48% at 1.5M).

Where it fails: 27 of the 37 misses in the 3M evaluation are targets **behind** the arm
(bearing > 90 deg), which need an over-the-top configuration. On targets in front, success
is 94% (157 of 167). So targets are now limited to `target_max_bearing_deg = 90` (all the
`SAFE_TARGETS` are within 25 deg of straight ahead); `--max-bearing -1` restores all-around
targets. Runs saved before this setting existed still evaluate with all-around targets.

Coverage gaps: sampling uniformly random *joint* angles spreads targets unevenly in space.
Close-in, high points (e.g. 18 cm out, 22 cm up) were rare, and the joint-sampled policy
stalled about 4.5 cm short there even though an IK solution exists. `WorkspaceSampler`
(`target_sampling = "workspace"`, the default) instead picks a 3 cm cell of the reachable
space uniformly, then a reachable point inside it. Tested on 300 targets from *either*
distribution, the joint-sampled policy gets about 92% and the workspace-sampled one 100%.
It also reaches that old point to 0.1 cm. That safe target was still moved to
(0.24, 0.05, 0.22), further inside the workspace. Each result is a single training run
(one seed).

## Evaluate

```powershell
.\.venv\Scripts\python.exe evaluate.py --model runs\ppo_reach_ws\best_model.zip --episodes 100
```
Prints success rate (final tip distance < 2 cm), mean/median/p90 final distance, and writes
`runs\ppo_reach_ws\rollouts.mp4` with the first 3 episodes. `--dr` evaluates under randomization.

## Environment

| | |
|---|---|
| Control | 20 Hz (25 x 2 ms physics steps), 100 steps = 5 s episodes, no early termination |
| Action | 5 values in [-1, 1] -> delta joint-position target, <= 0.05 rad (2.9 deg) per step, added to the previous *target* and clamped to the joint limits minus 5 deg. Gripper held fixed. |
| Observation (24) | joint angles (5), finite-difference joint velocity (5), tip position (3), target (3), tip-to-target vector x5 (3), previous action (5) |
| Reward | `-d + 0.5 * [d < 2 cm] + (1 - tanh(d / 5 cm)) - 0.01 * |a|^2`, d = tip-to-target distance [m] |
| Targets | spread evenly over the reachable space (3 cm cells of a 300k-point FK pool); a point is kept if tip is >= 5 cm above the table, >= 12 cm from the base axis, within 90 deg of straight ahead (+x), and the arm is collision-free |
| Tip point | the model's own `gripperframe` site: between the jaws, ~6 mm behind the jaw tips |

The observation is built only from joint angles (FK and finite differences are computed
from them), so the real arm, where LeRobot reports positions only, gets exactly the same
inputs. `build_observation()` and `apply_action()` are shared by the env and `run_real.py`.

**Domain randomization** (`DomainRandomization`, off by default; each item independent):
joint damping, armature, friction loss, actuator gain (kp and kv scaled together; values
below 1 model your 7.4 V servos on a 5 V supply), Gaussian joint-angle noise, and 0..N steps
of action latency. `DomainRandomization.sim2real()` is a starting preset; `--dr` uses it.

## Second task: push a block (sim only, in progress)

`--task push` trains the same arm to push a 4 cm, 50 g cube onto a goal disc on the table
(`so101_rl/push_env.py`, `assets/so101/push_scene.xml`). Same 5-joint action interface and
20 Hz. Episodes are 200 steps (10 s), and success means the block centre ends within 3 cm of
the goal. The observation (31 values) adds block position, block velocity and goal. The reward
pays for the tip being at a push point behind the block (on the side away from the goal), for
each step's progress of the block toward the goal, and for being close to the goal.
`run_real.py` refuses push models.

Overnight run (about 7 h at ~1,200 steps/s):
```powershell
.\.venv\Scripts\python.exe train.py --task push --run-name push_overnight --timesteps 30000000 --eval-freq 250000 --checkpoint-freq 1000000
```
Then:
```powershell
.\.venv\Scripts\python.exe evaluate.py --model runs\push_overnight\best_model.zip --episodes 100
```

Trials so far (1.5M steps each, about 20 min):

| Reward | Success | What the policy did |
|---|---|---|
| near-block reach term + distance to goal | 6% | touched the block but pushed it in random directions |
| push point + per-step progress (current) | 8% (best periodic eval 16% at 1.1M) | moves the block further (7.6 cm) but not yet consistently toward the goal |

Pushing usually needs tens of millions of steps with PPO, so these trials only show the
learning signal exists. They don't predict where it will end.

## Real arm: `run_real.py`

**Default is a dry run**: it connects, reads the arm, and prints every command it would send.
It sends nothing. Nothing in this project has been run against the real arm.

1. Mapping check (read-only, no commands sent; writes `mapping_check.png`):
   ```powershell
   .\.venv\Scripts\python.exe run_real.py --check-mapping
   ```
   With `--limp` torque is switched off so you can move the arm by hand: **support it first**.
2. Dry run of the policy:
   ```powershell
   .\.venv\Scripts\python.exe run_real.py --model runs\ppo_reach_ws\best_model.zip
   ```
3. Live (asks you to type `YES`):
   ```powershell
   .\.venv\Scripts\python.exe run_real.py --model runs\ppo_reach_ws\best_model.zip --live --targets 0
   ```

Sequence: read the start pose, move slowly to the home pose, then for each of the fixed
`SAFE_TARGETS` run the policy for `--steps` ticks and go back to home, finally return to the
start pose and disconnect (torque off, as in your other scripts).

Safety layers every tick: action clipped to [-1, 1]; joint target clamped to the sim limits
**and** your calibrated range minus 3 deg (LeRobot does not clamp in degrees mode); per-step
change <= `--max-step-deg` (default 1.5 deg = 30 deg/s at 20 Hz, hard cap 3 deg); LeRobot's
`max_relative_target` (default 5 deg vs. the measured position); targets only from the fixed
list. **Ctrl+C** stops the policy and slowly returns to the start pose; **Ctrl+C again**
holds the current pose and exits with torque left ON. `--fake` runs the whole script
against a simulated arm.

### Units and mapping (LeRobot 0.4.4)

* `SO101FollowerConfig.use_degrees` defaults to **True** in 0.4.4. Your scripts do not set
  it, so the values they print and `waypoints.json` are **degrees**, not -100..100
  (e.g. shoulder_pan -16.3956 is exactly raw tick 1861). The "-100..100" comment in
  `keyboard_drive.py` is out of date; its +/-100 clamp is really +/-100 deg.
* Degrees mode: `deg = (raw - mid) * 360 / 4095`, `mid = (range_min + range_max) / 2` from
  the calibration file. It ignores `drive_mode` and does not clamp on write.
* Mapping used: `q_mujoco = sign * deg2rad(lerobot_deg - offset)`, with **sign = +1 and
  offset = 0 for every joint**. Evidence: (1) the official `so101_new_calib.xml` puts zero at
  mid-range, the same reference LeRobot's degrees use; (2) LeRobot's own SO-101 kinematics
  (`lerobot/model/kinematics.py`) feeds degrees into the same model with plain `deg2rad`;
  (3) your recorded calibrated half-ranges (lift 105, elbow 98, wrist_flex 99 deg) match the
  model's (100, 97, 95 deg); (4) all 6 of your `waypoints.json` poses land inside the model's
  limits, and flipping shoulder_lift drives every one of them into the floor.
* `--range-mode` supports `use_degrees=False`; tests verify both modes against LeRobot's
  actual `_normalize`/`_unnormalize` and the int-tick quantization (< 1 tick error).

### Assumptions to verify on the hardware

1. **Joint directions** for elbow_flex and wrist_flex (and pan/roll). Run `--check-mapping`
   and compare `mapping_check.png` with the arm. If a joint bends the wrong way, pass
   e.g. `--sign elbow_flex=-1` (and use the same for the policy run).
2. **Zero**: at the pose where the upper arm is vertical and the forearm points straight
   forward horizontally, `--check-mapping` should print about 0 deg for lift/elbow/wrist_flex.
   A few degrees off is expected from calibration; much more means an offset is needed.
3. Your folded rest pose shows a few mm of overlap between collision hulls at the elbow stop
   in sim; that is a hull approximation, not a mapping error.
4. `shoulder_pan` was calibrated over the full 0..4095 ticks, so its calibrated range gives
   no protection; the sim limit (+/-105 deg after margins) is what clamps it.
5. Servo strength: 5 V supply means less torque and speed than the sim's STS3215 model.
   If tracking lags, retrain with `--dr`.
6. Connecting through LeRobot enables torque and rewrites PID/acceleration registers
   (normal LeRobot behaviour, same as your other scripts).
