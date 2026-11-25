"""Run a trained reach policy on the real SO-101 follower (LeRobot 0.4.4).

DEFAULT IS DRY RUN: it connects, reads the arm, and prints every command it WOULD
send, but sends nothing. Moving the arm requires --live plus typing YES.

Examples (PowerShell, from the project folder):
  # read-only: compare the real pose with the sim pose (writes mapping_check.png)
  .\\.venv\\Scripts\\python.exe run_real.py --check-mapping
  # dry run of the policy (default)
  .\\.venv\\Scripts\\python.exe run_real.py --model runs\\ppo_reach\\best_model.zip
  # no hardware at all: simulated arm, for testing this script
  .\\.venv\\Scripts\\python.exe run_real.py --model runs\\ppo_reach\\best_model.zip --fake
  # real motion
  .\\.venv\\Scripts\\python.exe run_real.py --model runs\\ppo_reach\\best_model.zip --live

Safety layers (in order, every control tick):
  1. policy action clipped to [-1, 1] and scaled by the training max_delta,
  2. clamp to the sim training limits AND the calibrated range minus a margin,
  3. per-step change limited to --max-step-deg (default 1.5 deg, hard cap 3 deg),
  4. LeRobot's own max_relative_target (goal vs present position, default 5 deg),
  5. targets come only from the fixed SAFE_TARGETS list below.
Ctrl+C: stop the policy and slowly return to the start pose. Ctrl+C again: hold
the current pose and exit WITHOUT disabling torque (power the arm off yourself).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from so101_rl.config import EnvConfig
from so101_rl.env import SO101ReachEnv, apply_action, build_observation
from so101_rl.kinematics import ARM_JOINTS, Kinematics
from so101_rl.mapping import DEFAULT_CALIBRATION, MotorCalib, SimRealMapping, load_calibration
from so101_rl.safety import SafetyFilter, interpolate

# Fixed, conservative targets in the sim/base frame [m]: in front of the arm, well above
# the table, well inside the workspace. tests/test_real_script.py checks each is reachable
# (collision-free IK solution inside the safe joint box).
SAFE_TARGETS: list[tuple[float, float, float]] = [
    (0.25, 0.00, 0.15),
    (0.22, 0.10, 0.12),
    (0.22, -0.10, 0.12),
    (0.30, 0.00, 0.20),
    # Was (0.18, 0.05, 0.22): reachable, but in a close-in, high pocket that random joint
    # sampling almost never produces (0 of 3000 training targets within 3 cm), so the policy
    # stalled 4.5 cm short with the elbow on its limit. 6 cm further out it is well covered.
    (0.24, 0.05, 0.22),
]
HARD_MAX_STEP_DEG = 3.0
CALIB_MARGIN_DEG = 3.0


# ----------------------------------------------------------------------------- arms
class LeRobotArm:
    """Thin wrapper over lerobot SO101Follower. `live=False` never writes goal positions."""

    def __init__(self, port: str, robot_id: str, use_degrees: bool, max_relative_target: float, live: bool):
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        cfg = SO101FollowerConfig(
            port=port,
            id=robot_id,
            use_degrees=use_degrees,
            max_relative_target=max_relative_target,
            disable_torque_on_disconnect=True,
        )
        self.robot = SO101Follower(cfg)
        self.live = live
        # calibrate=False: never start an interactive calibration from this script.
        self.robot.connect(calibrate=False)
        if not self.robot.is_calibrated:
            self.robot.disconnect()
            raise RuntimeError("Motor calibration does not match the calibration file. Run lerobot-calibrate first.")

    def calibration(self) -> dict[str, MotorCalib]:
        return {
            k: MotorCalib(c.id, c.drive_mode, c.homing_offset, c.range_min, c.range_max)
            for k, c in self.robot.calibration.items()
        }

    def read(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.robot.get_observation().items() if k.endswith(".pos")}

    def send(self, action: dict[str, float]) -> None:
        if not self.live:
            raise RuntimeError("send() called on a non-live arm")  # defensive: dry run never sends
        self.robot.send_action(action)

    def set_torque(self, on: bool) -> None:
        if on:
            self.robot.bus.enable_torque()
        else:
            self.robot.bus.disable_torque()

    def close(self, disable_torque: bool = True) -> None:
        self.robot.bus.disconnect(disable_torque)


class FakeArm:
    """Simulated arm with the LeRobot interface (MuJoCo physics), for testing without hardware."""

    def __init__(self, mapping: SimRealMapping, start_q: np.ndarray | None = None):
        self.env = SO101ReachEnv()
        self.env.reset(seed=0)
        self.mapping = mapping
        if start_q is not None:
            self.env.data.qpos[self.env.qpos_adr] = start_q
            self.env.data.ctrl[self.env.act_ids] = start_q
        self.live = True
        self.sent: list[dict[str, float]] = []

    def read(self) -> dict[str, float]:
        q = self.env.data.qpos[self.env.qpos_adr]
        return self.mapping.sim_to_real(q) | {"gripper.pos": 0.0}

    def send(self, action: dict[str, float]) -> None:
        self.sent.append(dict(action))
        q = self.mapping.real_to_sim(action)
        self.env.data.ctrl[self.env.act_ids] = q
        import mujoco

        mujoco.mj_step(self.env.model, self.env.data, nstep=self.env.n_substeps)

    def set_torque(self, on: bool) -> None:
        pass

    def close(self, disable_torque: bool = True) -> None:
        pass


# ----------------------------------------------------------------------------- runner
class Runner:
    def __init__(self, arm, mapping: SimRealMapping, cfg: EnvConfig, max_step: float, dry_run: bool, log=print):
        self.arm, self.mapping, self.cfg, self.dry_run, self.log = arm, mapping, cfg, dry_run, log
        self.kin = Kinematics(gripper_q=cfg.gripper_q)
        lim = self.kin.limits
        self.safety = SafetyFilter(
            lim[:, 0] + cfg.limit_margin,
            lim[:, 1] - cfg.limit_margin,
            mapping.calibrated_limits_rad(),
            margin=np.radians(CALIB_MARGIN_DEG),
            max_step=max_step,
        )
        self.dt = 1.0 / cfg.control_hz
        start = arm.read()
        self.gripper_hold = start.get("gripper.pos")
        self.q_start = mapping.real_to_sim(start)
        self.q_cmd = self.q_start.copy()  # last command (in dry run: assumed reached)
        self.q_obs_prev = self.q_start.copy()
        self._next_t = time.perf_counter()
        if not self.safety.contains(self.q_start):
            self.log(
                "NOTE: start pose is outside the safe box; the first moves will bring it inside "
                f"(max {np.degrees(max_step):.1f} deg per step)."
            )

    def read_q(self) -> np.ndarray:
        if self.dry_run:
            return self.q_cmd.copy()
        return self.mapping.real_to_sim(self.arm.read())

    def command(self, q_desired: np.ndarray) -> np.ndarray:
        q_safe = self.safety(q_desired, self.q_cmd)
        action = self.mapping.sim_to_real(q_safe)
        if self.gripper_hold is not None:
            action["gripper.pos"] = self.gripper_hold
        if self.dry_run:
            self.log("DRY-RUN would send: " + "  ".join(f"{k.split('.')[0]}={v:7.2f}" for k, v in action.items()))
        else:
            self.arm.send(action)
        self.q_cmd = q_safe
        self._sleep()
        return q_safe

    def _sleep(self) -> None:
        self._next_t += self.dt
        delay = self._next_t - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        else:
            self._next_t = time.perf_counter()

    def move_to(self, q_goal: np.ndarray, label: str) -> None:
        q_goal = np.clip(q_goal, self.safety.low, self.safety.high)
        self.log(f"--> moving to {label}: {np.degrees(q_goal).round(1)} deg (sim frame)")
        for q in interpolate(self.q_cmd, q_goal, self.safety.max_step):
            self.command(q)

    def run_policy(self, policy, target: np.ndarray, steps: int) -> float:
        self.log(f"--> target {np.round(target, 3)} m for {steps} steps")
        q_target = self.q_cmd.copy()
        prev_action = np.zeros(len(ARM_JOINTS))
        q_prev = self.read_q()
        dist = float("nan")
        for t in range(steps):
            q_obs = self.read_q()
            ee = self.kin.fk(q_obs)
            obs = build_observation(q_obs, q_prev, self.dt, ee, target, prev_action)
            q_prev = q_obs
            action, _ = policy.predict(obs, deterministic=True)
            action = np.clip(action, -1.0, 1.0)
            q_target = apply_action(q_target, action, self.cfg.max_delta, self.safety.low, self.safety.high)
            q_target = self.command(q_target)  # rate-limited command is the new integrator state
            prev_action = action
            dist = float(np.linalg.norm(self.kin.fk(self.read_q()) - target))
            if t % 10 == 0 or t == steps - 1:
                self.log(f"    step {t:3d}  est. tip distance {dist * 100:5.1f} cm")
        return dist


def find_env_config(model_path: Path) -> EnvConfig:
    for d in (model_path.parent, model_path.parent.parent):
        p = d / "env_config.json"
        if p.exists():
            return EnvConfig.from_dict(json.loads(p.read_text()))
    print("WARNING: env_config.json not found next to the model; using EnvConfig defaults.")
    return EnvConfig()


def check_mapping(arm, mapping: SimRealMapping, seconds: float, image: Path) -> None:
    """Read-only: print real joint values, sim angles, and sim tip position; save a render."""
    import imageio
    import mujoco

    kin = Kinematics()
    t_end = time.time() + seconds
    q = None
    print("Move the arm by hand (only if you used --limp) and compare with the sim. Ctrl+C to stop.")
    try:
        while time.time() < t_end:
            obs = arm.read()
            q = mapping.real_to_sim(obs)
            tip = kin.fk(q)
            print(
                "real: "
                + " ".join(f"{j[:6]}={obs[j + '.pos']:7.2f}" for j in ARM_JOINTS)
                + " | sim deg: "
                + " ".join(f"{d:7.1f}" for d in np.degrees(q))
                + f" | tip xyz [cm]: {np.round(tip * 100, 1)}"
            )
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    if q is not None:
        env = SO101ReachEnv(render_mode="rgb_array")
        env.reset(seed=0, options={"target_pos": kin.fk(q)})
        env.data.qpos[env.qpos_adr] = q
        mujoco.mj_forward(env.model, env.data)
        imageio.imwrite(image, env.render())
        env.close()
        print(f"Saved sim render of the last pose to {image}. Compare it with the real arm.")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, help="trained PPO .zip")
    p.add_argument("--port", default="COM20")
    p.add_argument("--robot-id", default="my_follower")
    p.add_argument("--calibration", type=Path, default=None, help=f"default: {DEFAULT_CALIBRATION}")
    p.add_argument("--range-mode", action="store_true", help="use LeRobot use_degrees=False (-100..100)")
    p.add_argument("--live", action="store_true", help="actually move the arm (default: dry run)")
    p.add_argument("--fake", action="store_true", help="simulated arm, no hardware")
    p.add_argument("--check-mapping", action="store_true", help="read-only mapping check")
    p.add_argument("--limp", action="store_true", help="with --check-mapping: torque off (SUPPORT THE ARM)")
    p.add_argument("--seconds", type=float, default=30.0, help="--check-mapping duration")
    p.add_argument("--targets", type=int, nargs="*", default=None, help="indices into SAFE_TARGETS (default: all)")
    p.add_argument("--steps", type=int, default=100, help="policy steps per target")
    p.add_argument("--max-step-deg", type=float, default=1.5, help=f"per-step joint change (<= {HARD_MAX_STEP_DEG})")
    p.add_argument("--max-relative-deg", type=float, default=5.0, help="LeRobot max_relative_target (deg mode)")
    p.add_argument("--sign", action="append", default=[], metavar="JOINT=+1|-1", help="override a joint sign")
    return p.parse_args(argv)


def range_units_per_degree(cal: dict[str, MotorCalib]) -> float:
    """Smallest RANGE_M100_100 units per degree over the arm joints (conservative)."""
    return min(200 / (cal[j].range_max - cal[j].range_min) * 4095 / 360 for j in ARM_JOINTS)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.max_step_deg <= 0 or args.max_step_deg > HARD_MAX_STEP_DEG:
        print(f"--max-step-deg must be in (0, {HARD_MAX_STEP_DEG}]")
        return 2
    if args.live and args.fake:
        print("--live and --fake are exclusive")
        return 2
    if not args.check_mapping and args.model is None:
        print("--model is required (or use --check-mapping)")
        return 2
    signs = {}
    for s in args.sign:
        j, v = s.split("=")
        signs[j] = int(v)
    use_degrees = not args.range_mode
    targets = [np.array(SAFE_TARGETS[i]) for i in (args.targets if args.targets else range(len(SAFE_TARGETS)))]

    dry_run = not args.live and not args.fake
    print(f"MODE: {'FAKE (simulated arm)' if args.fake else 'LIVE - THE ARM WILL MOVE' if args.live else 'DRY RUN'}")
    if args.live:
        print("Clear the workspace, keep a hand near the power switch.")
        if input("Type YES to move the real arm: ").strip() != "YES":
            print("Aborted.")
            return 1

    # --- arm + mapping
    if args.fake:
        cal = load_calibration(args.calibration or DEFAULT_CALIBRATION)
        mapping = SimRealMapping(cal, use_degrees=use_degrees, signs=signs)
        arm = FakeArm(mapping, start_q=np.array([0.0, -1.2, 1.2, 0.8, 0.0]))
    else:
        max_rel = args.max_relative_deg
        if not use_degrees:
            max_rel *= range_units_per_degree(load_calibration(args.calibration or DEFAULT_CALIBRATION))
        arm = LeRobotArm(args.port, args.robot_id, use_degrees, max_rel, live=args.live)
        cal = arm.calibration() if args.calibration is None else load_calibration(args.calibration)
        mapping = SimRealMapping(cal, use_degrees=use_degrees, signs=signs)

    if args.check_mapping:
        try:
            if args.limp and not args.fake:
                input("Torque will be DISABLED. Support the arm, then press Enter...")
                arm.set_torque(False)
            check_mapping(arm, mapping, args.seconds, Path("mapping_check.png"))
        finally:
            arm.close(disable_torque=False if not args.limp else True)
        return 0

    from stable_baselines3 import PPO

    policy = PPO.load(str(args.model), device="cpu")
    cfg = find_env_config(args.model)
    runner = Runner(arm, mapping, cfg, max_step=np.radians(args.max_step_deg), dry_run=dry_run)
    print(f"Start pose (sim deg): {np.degrees(runner.q_start).round(1)}  control {cfg.control_hz:.0f} Hz")
    home = np.asarray(cfg.home_qpos, float)

    disable_torque_at_end = True
    try:
        runner.move_to(home, "home")
        for tgt in targets:
            d = runner.run_policy(policy, tgt, args.steps)
            print(f"    final est. distance {d * 100:.1f} cm")
            runner.move_to(home, "home")
        runner.move_to(runner.q_start, "start pose")
    except KeyboardInterrupt:
        print("\nCtrl+C: stopping policy, returning slowly to the start pose (Ctrl+C again to hold here).")
        try:
            runner.move_to(runner.q_start, "start pose")
        except KeyboardInterrupt:
            print("\nSecond Ctrl+C: holding current pose; torque stays ON. Power the arm off when ready.")
            if not dry_run:
                try:
                    arm.send(mapping.sim_to_real(runner.read_q()) | {"gripper.pos": runner.gripper_hold})
                except Exception as e:  # noqa: BLE001 - best effort hold
                    print(f"hold command failed: {e}")
            disable_torque_at_end = False
    finally:
        arm.close(disable_torque=disable_torque_at_end)
        print("Disconnected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
