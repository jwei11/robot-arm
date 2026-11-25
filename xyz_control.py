"""Move the SO-101 gripper in x, y, z (centimetres) instead of joint angles.

Run with the project venv (it has MuJoCo, which provides the arm's kinematics):

    .\\.venv\\Scripts\\python.exe xyz_control.py
    .\\.venv\\Scripts\\python.exe xyz_control.py --check     # read-only, torque off: where is the tip?

Frame (base of the arm, from the official SO-101 model):
    x = forward, y = left, z = up, origin at the base. Pitch -90 = gripper pointing straight down.

Start: torque is OFF. You pose the arm by hand and press Enter; it locks there.
The script never makes an automatic move on its own.

Keys (hold to keep moving; terminal must have focus):
    A / D   turn the base left / right (nothing else moves)
    W / S   reach out / pull in        R / F   up / down      (base stays still, gripper angle held)
    T / G   tilt gripper up / down    Q / E   wrist roll             Y / H   gripper open / close
    1 2 3   step 0.5 / 1 / 2 cm       c       type a coordinate to go to
    space   stop where it is          z       back to where you locked it    p   print position
    Esc     stop and release (Ctrl+C does the same)

Safety:
  - no automatic moves at start or exit; you place the arm, and it goes limp when you quit,
  - every move (jog, typed coordinate, return) is checked step by step: inside the joint
    limits (sim limits AND calibrated range minus 3 deg), tip >= 2 cm above the table, no
    self/table collision; any single move bigger than 45 deg on a joint is refused,
  - joints move at most --speed deg/s (default 25); typed moves at 20 deg/s,
  - bump-to-stop (arm_guard.py): a joint lagging > 3.5 deg beyond normal freezes the arm;
    a second bump within 3 s, or a joint still being pushed after freezing, RELEASES torque,
  - losing contact with the motors stops the script and tells you to unplug the power.
"""

from __future__ import annotations

import argparse
import math
import msvcrt
import os
import shutil
import signal
import time

import numpy as np

from arm_config import JOINT_SIGNS, PORT, ROBOT_ID
from arm_guard import BumpGuard, read_telemetry
from so101_rl.config import EnvConfig
from so101_rl.kinematics import ARM_JOINTS, Kinematics
from so101_rl.mapping import SimRealMapping
from so101_rl.safety import SafetyFilter

HZ = 50
CALIB_MARGIN_DEG = 3.0
MIN_Z = 0.02  # m, never put the tip lower than this
IK_TOL = 0.003  # m, max position error to accept a target
PITCH_W = 0.05  # m per rad: how much pitch error "costs" vs position error
MAX_JUMP_DEG = 45.0  # largest single joint change allowed in one move
MAX_JOG_DEG_PER_CM = 12.0  # a jog step needing more joint motion than this per cm = near a singularity
MAX_LOCK_Z = 0.40  # m, above this the arm is nearly stretched straight up (awkward to steer)
MAX_LEAD_DEG = 2.0  # ignore jog keys while the arm is still this far behind its goal
MIN_RADIUS = 0.12  # m, keep the tip this far from the base's vertical axis (straight-up = singular)

REACH_KEYS = {"w": ("r", +1), "s": ("r", -1), "r": ("z", +1), "f": ("z", -1)}  # base stays still
PAN_KEYS = {"a": -1, "d": +1}  # in the model, positive pan swings the arm to the right (-y)
MAX_PAN_STEP_DEG = 6.0
MAX_TILT_JUMP_DEG = 15.0  # a 3 deg gripper tilt needing more than this on a joint = awkward
STEPS_CM = {"1": 0.5, "2": 1.0, "3": 2.0}

interrupts = 0


def on_sigint(signum, frame):
    global interrupts
    interrupts += 1


# ------------------------------------------------------------------ kinematics
class ArmIK:
    """Position + gripper pitch IK over pan/lift/elbow/wrist_flex (wrist_roll held)."""

    def __init__(self, low, high):
        self.kin = Kinematics()
        self.low, self.high = np.asarray(low), np.asarray(high)
        self.wf_id = self.kin.joint_ids[3]

    def pose(self, q):
        """(tip xyz [m], pitch [rad]) for joint vector q (5,)."""
        tip = self.kin.fk(q)
        wrist = self.kin.data.xanchor[self.wf_id].copy()
        a = tip - wrist
        return tip, math.atan2(a[2], math.hypot(a[0], a[1]))

    def solve(self, q_init, target, pitch, iters=40):
        """Damped least squares from q_init. pitch=None leaves the gripper angle free.
        Returns (q, position error [m], pitch error [rad])."""
        q = np.array(q_init, float)
        idx = [0, 1, 2, 3]
        eps = 1e-5
        w = 0.0 if pitch is None else PITCH_W
        for _ in range(iters):
            tip, p = self.pose(q)
            r = np.r_[target - tip, w * ((p if pitch is None else pitch) - p)]
            if np.linalg.norm(r[:3]) < 2e-4 and abs(r[3]) < PITCH_W * 0.005:
                break
            J = np.zeros((4, 4))
            for c, i in enumerate(idx):
                dq = q.copy()
                dq[i] += eps
                t2, p2 = self.pose(dq)
                J[:3, c] = (t2 - tip) / eps
                J[3, c] = w * (p2 - p) / eps
            step = J.T @ np.linalg.solve(J @ J.T + 1e-5 * np.eye(4), r)
            step = np.clip(step, -0.2, 0.2)
            q[idx] = np.clip(q[idx] + step, self.low[idx], self.high[idx])
        tip, p = self.pose(q)
        return q, float(np.linalg.norm(target - tip)), 0.0 if pitch is None else float(pitch - p)

    def solve_planar(self, q_init, r_t, z_t, pitch, iters=60):
        """Base held still: move only lift/elbow/wrist_flex so the tip is r_t out from the base
        axis, z_t high, at gripper angle `pitch`. Returns (q, position error [m], pitch error [rad])."""
        q = np.array(q_init, float)
        idx = [1, 2, 3]
        eps = 1e-5

        def resid(qq):
            tip, p = self.pose(qq)
            return np.array([r_t - math.hypot(tip[0], tip[1]), z_t - tip[2], PITCH_W * (pitch - p)])

        for _ in range(iters):
            r = resid(q)
            if np.linalg.norm(r[:2]) < 2e-4 and abs(r[2]) < PITCH_W * 0.005:
                break
            J = np.zeros((3, 3))
            for c, i in enumerate(idx):
                dq = q.copy()
                dq[i] += eps
                J[:, c] = (r - resid(dq)) / eps
            step = J.T @ np.linalg.solve(J @ J.T + 1e-5 * np.eye(3), r)
            q[idx] = np.clip(q[idx] + np.clip(step, -0.2, 0.2), self.low[idx], self.high[idx])
        r = resid(q)
        return q, float(np.linalg.norm(r[:2])), float(r[2] / PITCH_W)

    def rzp(self, q):
        """(reach out from the base axis [m], height [m], gripper pitch [rad])."""
        tip, p = self.pose(q)
        return (math.hypot(tip[0], tip[1]), float(tip[2]), p)

    def collides(self, q):
        return self.kin.in_collision(q)


class ReleaseTorque(Exception):
    """Something keeps pushing back: stop commanding and let the arm go limp."""


# ------------------------------------------------------------------ helpers
def status_line(tip, pitch, q, grip, step_cm, msg):
    # message first: on a narrow terminal the end of the line gets cut off
    s = (
        f"{msg + ' | ' if msg else ''}"
        f"x{tip[0] * 100:5.1f} y{tip[1] * 100:5.1f} z{tip[2] * 100:5.1f}cm "
        f"pitch{math.degrees(pitch):4.0f} roll{math.degrees(q[4]):4.0f} grip{grip:3.0f} step {step_cm}cm"
    )
    width = shutil.get_terminal_size((100, 20)).columns - 1
    print(f"\r\x1b[2K{s[:width]}", end="", flush=True)


def path_problem(ik, safety, q_from, q_to):
    """None if every step from q_from to q_to is inside limits, above the table and collision-free."""
    jump = float(np.degrees(np.max(np.abs(q_to - q_from))))
    if jump > MAX_JUMP_DEG:
        return f"needs a {jump:.0f} deg joint move (max {MAX_JUMP_DEG:.0f}) - go in smaller steps"
    n = max(2, int(jump / 2))
    for k in range(1, n + 1):
        q = q_from + (q_to - q_from) * k / n
        if not safety.contains(q):
            return "path leaves the joint limits"
        if ik.pose(q)[0][2] < MIN_Z:
            return "path goes too close to the table"
        if ik.collides(q):
            return "path collides"
    return None


def jog_step(ik, safety, q_goal, rzp, k, step_cm):
    """One key press of jogging. Returns (new q_goal, new (reach, height, pitch), message);
    the goal is unchanged when the move is refused."""
    q_new, new_rzp = q_goal.copy(), rzp
    if k in PAN_KEYS:
        # turn the base only
        d = min(math.radians(MAX_PAN_STEP_DEG), step_cm / 100 / max(rzp[0], MIN_RADIUS))
        q_new[0] += PAN_KEYS[k] * d
        err = 0.0
        jump_lim = MAX_PAN_STEP_DEG + 0.1
    elif k in "qe":
        q_new[4] += math.radians(4) * (1 if k == "q" else -1)
        err = 0.0
        jump_lim = 4.1
    else:
        r_t, z_t, p_t = rzp
        if k in REACH_KEYS:
            which, sign = REACH_KEYS[k]
            if which == "r":
                r_t += sign * step_cm / 100
            else:
                z_t += sign * step_cm / 100
            jump_lim = MAX_JOG_DEG_PER_CM * step_cm
        else:
            p_t += math.radians(3) * (1 if k == "t" else -1)
            jump_lim = MAX_TILT_JUMP_DEG
        new_rzp = (r_t, z_t, p_t)
        q_new, err, perr = ik.solve_planar(q_goal, r_t, z_t, p_t)
        if abs(perr) > math.radians(1):
            err = max(err, IK_TOL * 2)
    jog_jump = float(np.degrees(np.max(np.abs(q_new - q_goal))))
    tip_new = ik.pose(q_new)[0]
    problem = None
    if err <= IK_TOL and jog_jump <= jump_lim:
        problem = path_problem(ik, safety, q_goal, q_new)
    if new_rzp[1] < MIN_Z or tip_new[2] < MIN_Z:
        return q_goal, rzp, "LIMIT: table"
    elif k == "r" and new_rzp[1] > MAX_LOCK_Z:
        return q_goal, rzp, f"LIMIT: {MAX_LOCK_Z * 100:.0f} cm high (arm nearly straight up)"
    elif new_rzp[0] < MIN_RADIUS:
        return q_goal, rzp, "LIMIT: too close to the base"
    elif err > IK_TOL:
        return q_goal, rzp, "LIMIT: out of reach at this gripper angle - tilt it with T/G"
    elif jog_jump > jump_lim:
        return q_goal, rzp, f"LIMIT: awkward ({jog_jump:.0f} deg for one step) - tilt with T/G or go another way"
    elif problem:
        return q_goal, rzp, f"LIMIT: {problem}"
    else:
        return q_new, new_rzp, ""


def main():
    global interrupts
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="read-only, torque off: print the tip position")
    ap.add_argument("--speed", type=float, default=25.0, help="max joint speed while jogging, deg/s (default 25)")
    ap.add_argument("--no-guard", action="store_true", help="disable bump-to-stop")
    args = ap.parse_args()
    os.system("")

    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    robot = SO101Follower(SO101FollowerConfig(port=PORT, id=ROBOT_ID, use_degrees=True))
    bus = robot.bus
    # Not lerobot's connect(): that re-enables torque at whatever goal the motors last had.
    # Here the goal is set to the present position first, and torque ends up OFF.
    bus.connect()
    if not bus.is_calibrated:
        bus.port_handler.closePort()
        raise SystemExit("Calibration mismatch - run lerobot-calibrate first.")
    bus.disable_torque()
    bus.sync_write("Goal_Position", bus.sync_read("Present_Position", normalize=False), normalize=False)
    robot.configure()  # servo gains; ends with torque on at goal == present (no motion)
    bus.disable_torque()

    mapping = SimRealMapping(robot.calibration, use_degrees=True, signs=JOINT_SIGNS)
    cfg = EnvConfig()
    kin_lim = Kinematics().limits
    max_step = math.radians(args.speed) / HZ
    safety = SafetyFilter(
        kin_lim[:, 0] + cfg.limit_margin,
        kin_lim[:, 1] - cfg.limit_margin,
        mapping.calibrated_limits_rad(),
        margin=math.radians(CALIB_MARGIN_DEG),
        max_step=max_step,
    )
    ik = ArmIK(safety.low, safety.high)
    dt = 1.0 / HZ

    def read_q():
        return mapping.real_to_sim(robot.get_observation())

    def close():
        try:
            bus.disable_torque(num_retry=3)
        except Exception as e:  # noqa: BLE001
            print(f"\nCould not switch torque off ({e}). UNPLUG THE ARM'S POWER.")
        try:
            bus.port_handler.closePort()
        except Exception:  # noqa: BLE001
            pass

    # ---------------- read-only check (torque stays off)
    if args.check:
        try:
            print("Read-only, torque off. Ctrl+C to stop.\n")
            while True:
                q = read_q()
                tip, p = ik.pose(q)
                print(
                    f"\rjoints deg {np.degrees(q).round(0)}  ->  tip x {tip[0] * 100:5.1f} "
                    f"y {tip[1] * 100:5.1f} z {tip[2] * 100:5.1f} cm  pitch {math.degrees(p):4.0f}   ",
                    end="",
                    flush=True,
                )
                time.sleep(0.2)
        except KeyboardInterrupt:
            print()
        finally:
            close()
        return

    guard = None if args.no_guard else BumpGuard(bus)
    grip = float(robot.get_observation()["gripper.pos"])
    bump_times: list[float] = []
    state = {}

    def send(q):
        robot.send_action(mapping.sim_to_real(q) | {"gripper.pos": grip})

    def handle_bump(j):
        """Freeze; if it happens again soon or a joint is still being pushed, release torque."""
        guard.freeze()
        print(f"\nBUMP - {guard.describe(j)}. Holding still.")
        now = time.perf_counter()
        bump_times[:] = [t for t in bump_times if now - t < 3.0] + [now]
        time.sleep(0.5)  # send nothing; let it settle
        tel = read_telemetry(bus, full=False)
        still = [k for k, d in tel.items() if k != "gripper" and abs(d["err_deg"]) > guard.dev_deg]
        if len(bump_times) >= 2:
            raise ReleaseTorque("bumped twice within 3 s")
        if still:
            raise ReleaseTorque(f"{still[0]} still being pushed after freezing")
        state["q_cmd"] = read_q()
        guard.reset()

    def run_path(q_to, speed_deg=20.0):
        """Smooth joint move along an already-checked path. False if a bump stopped it."""
        q0 = state["q_cmd"].copy()
        n = max(1, int(np.max(np.abs(q_to - q0)) / (math.radians(speed_deg) * dt)))
        for k in range(1, n + 1):
            s = 0.5 - 0.5 * math.cos(math.pi * k / n)
            state["q_cmd"] = q0 + (q_to - q0) * s
            send(state["q_cmd"])
            if guard and k % 2 == 0 and (j := guard.check()):
                handle_bump(j)
                return False
            time.sleep(dt)
        return True

    try:
        print(__doc__)
        print("Torque is OFF. Hold the arm and pose it by hand where you want to start:")
        print("gripper in front of the base and above the table.\n")
        while True:
            input("Hold it there and press Enter to lock it (nothing will move)...")
            q_lock = read_q()
            tip, _ = ik.pose(q_lock)
            problems = []
            out = [ARM_JOINTS[i] for i in range(5) if not (safety.low[i] <= q_lock[i] <= safety.high[i])]
            if out:
                problems.append(f"outside the safe joint range: {', '.join(out)}")
            if tip[2] < MIN_Z:
                problems.append("gripper too close to the table")
            if tip[2] > MAX_LOCK_Z:
                problems.append(f"gripper {tip[2] * 100:.0f} cm high - lower it below 40 cm")
            if math.hypot(tip[0], tip[1]) < MIN_RADIUS:
                problems.append(
                    f"gripper only {math.hypot(tip[0], tip[1]) * 100:.0f} cm out from the base - "
                    "bend the arm forward so it reaches out at least 12 cm"
                )
            if ik.collides(q_lock):
                problems.append("arm touching itself or the table")
            if not problems and all(jog_step(ik, safety, q_lock, ik.rzp(q_lock), k, 1.0)[2] for k in "wsrf"):
                problems.append("arm stretched too far to steer - bend the elbow more and bring it closer/lower")
            if not problems:
                break
            print(
                f"Can't start here ({'; '.join(problems)}). Tip is at x {tip[0] * 100:.0f} "
                f"y {tip[1] * 100:.0f} z {tip[2] * 100:.0f} cm. Adjust it and try again."
            )

        # lock in place: goal = present, then torque on -> no motion
        bus.sync_write("Goal_Position", bus.sync_read("Present_Position", normalize=False), normalize=False)
        bus.enable_torque()
        signal.signal(signal.SIGINT, on_sigint)
        state["q_cmd"] = q_lock.copy()
        q_goal = q_lock.copy()
        target, _ = ik.pose(q_goal)
        rzp = ik.rzp(q_goal)
        print(
            f"\nLocked at x {target[0] * 100:.1f} y {target[1] * 100:.1f} z {target[2] * 100:.1f} cm. "
            "You can let go. Drive with the keys; Esc to stop.\n"
        )
        if guard:
            guard.reset()
        step_cm = 1.0
        msg = "ready"
        last_draw = 0.0
        next_t = time.perf_counter()
        tick = 0
        quitting = False

        while not quitting:
            keys = ["\x1b"] if interrupts else []
            while msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in ("\x00", "\xe0"):
                    msvcrt.getwch()
                    continue
                keys.append(ch)

            behind = float(np.degrees(np.max(np.abs(q_goal - state["q_cmd"]))))
            for ch in keys:
                k = ch.lower()
                if k in REACH_KEYS or k in PAN_KEYS or k in "tgqe":
                    if behind > MAX_LEAD_DEG:
                        continue  # arm still catching up: drop key repeats so it stops when you let go
                if k in REACH_KEYS or k in PAN_KEYS or k in "tgqe":
                    q_goal, rzp, msg = jog_step(ik, safety, q_goal, rzp, k, step_cm)
                    continue
                elif k == "y":
                    grip = min(100.0, grip + 3)
                    continue
                elif k == "h":
                    grip = max(0.0, grip - 3)
                    continue
                elif k in STEPS_CM:
                    step_cm = STEPS_CM[k]
                    continue
                elif k == " ":
                    q_goal = state["q_cmd"].copy()
                    rzp = ik.rzp(q_goal)
                    msg = "stopped"
                    continue
                elif k in ("z", "c"):
                    if k == "z":
                        q_to = q_lock
                    else:
                        signal.signal(signal.SIGINT, signal.default_int_handler)
                        try:
                            raw = input("\ngo to x y z [cm] (e.g. 20 0 15): ").replace(",", " ").split()
                            goal = np.array([float(v) / 100 for v in raw[:3]])
                            if goal.shape != (3,):
                                raise ValueError
                        except (ValueError, KeyboardInterrupt, EOFError):
                            msg = "bad coordinate"
                            continue
                        finally:
                            signal.signal(signal.SIGINT, on_sigint)
                        if goal[2] < MIN_Z:
                            msg = "refused: below the table margin"
                            continue
                        q_to, err, _ = ik.solve(state["q_cmd"], goal, rzp[2], iters=200)
                        if err > IK_TOL or ik.collides(q_to):
                            q_to, err, _ = ik.solve(state["q_cmd"], goal, None, iters=200)
                        if err > IK_TOL:
                            msg = f"refused: can't reach ({err * 100:.1f} cm off)"
                            continue
                    problem = path_problem(ik, safety, state["q_cmd"], q_to)
                    if problem:
                        msg = f"refused: {problem}"
                        continue
                    run_path(q_to)
                    q_goal = state["q_cmd"].copy()
                    rzp = ik.rzp(q_goal)
                    msg = "back at lock pose" if k == "z" else "arrived"
                    continue
                elif k == "p":
                    tip, p = ik.pose(state["q_cmd"])
                    print(
                        f"\nx={tip[0] * 100:.1f} y={tip[1] * 100:.1f} z={tip[2] * 100:.1f} cm  "
                        f"pitch={math.degrees(p):.0f} deg  joints(deg)={np.degrees(state['q_cmd']).round(1).tolist()}"
                    )
                    continue
                elif ch == "\x1b":
                    quitting = True
                    break

            if quitting:
                break

            # rate-limited step toward the goal (the goal is always a checked, in-limits pose)
            state["q_cmd"] = state["q_cmd"] + np.clip(q_goal - state["q_cmd"], -max_step, max_step)
            send(state["q_cmd"])
            tick += 1
            if guard and tick % 2 == 0 and (j := guard.check()):
                handle_bump(j)
                q_goal = state["q_cmd"].copy()
                rzp = ik.rzp(q_goal)
                msg = f"BUMP {j} - holding"

            now = time.perf_counter()
            if now - last_draw > 0.1:
                tip, p = ik.pose(state["q_cmd"])
                status_line(tip, p, state["q_cmd"], grip, step_cm, msg)
                last_draw = now
            next_t += dt
            wait = next_t - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            else:
                next_t = time.perf_counter()

        # normal exit: no automatic moves, just release
        signal.signal(signal.SIGINT, signal.default_int_handler)
        print("\n\nStopped and holding. Torque will be released and the arm goes limp.")
        try:
            input("Support the arm, then press Enter...")
        except (KeyboardInterrupt, EOFError):
            pass
        close()
    except ReleaseTorque as e:
        print(f"\n\nSAFETY STOP ({e}): torque released - the arm is limp. Support it.")
        close()
    except ConnectionError as e:
        print(f"\n\nLOST CONTACT WITH THE MOTORS: {e}\nUNPLUG THE ARM'S POWER before touching it.")
        close()
    except BaseException:
        close()
        raise


if __name__ == "__main__":
    main()
