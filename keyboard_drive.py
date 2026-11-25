"""Drive the SO-101 follower joint-by-joint from the keyboard (Windows).

Hold a key to move a joint; release to stop. Keys only register while this
terminal has focus.

    Joint           +    -
    shoulder_pan    Q    A
    shoulder_lift   W    S
    elbow_flex      E    D
    wrist_flex      R    F
    wrist_roll      T    G
    gripper         Y    H   (Y opens, H closes)

    1 / 2 / 3   speed: slow / medium / fast
    space       stop: freeze where the arm is right now
    z           go back to the start pose
    x           toggle torque (limp, so you can reposition by hand)
    p           print the current pose (copy it for scripts)
    Esc         return to start pose and quit
"""

import msvcrt
import os
import shutil
import signal
import time

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

PORT = "COM20"
ROBOT_ID = "my_follower"
HZ = 50

KEYMAP = {
    "q": ("shoulder_pan.pos", +1), "a": ("shoulder_pan.pos", -1),
    "w": ("shoulder_lift.pos", +1), "s": ("shoulder_lift.pos", -1),
    "e": ("elbow_flex.pos", +1), "d": ("elbow_flex.pos", -1),
    "r": ("wrist_flex.pos", +1), "f": ("wrist_flex.pos", -1),
    "t": ("wrist_roll.pos", +1), "g": ("wrist_roll.pos", -1),
    "y": ("gripper.pos", +1), "h": ("gripper.pos", -1),
}
# (units per key-repeat event, max joint speed units/s)
SPEEDS = {"1": ("slow", 1.0, 30.0), "2": ("medium", 2.5, 60.0), "3": ("fast", 5.0, 120.0)}

SHORT = {"shoulder_pan.pos": "pan", "shoulder_lift.pos": "lift", "elbow_flex.pos": "elb",
         "wrist_flex.pos": "wf", "wrist_roll.pos": "wr", "gripper.pos": "grip"}

# Ctrl+C: first press = ease back to start and quit, second press = quit immediately.
interrupts = 0


def on_sigint(signum, frame):
    global interrupts
    interrupts += 1


def read_pose(robot):
    return {k: float(v) for k, v in robot.get_observation().items() if k.endswith(".pos")}


def make_limits(start):
    """Calibrated range is -100..100 (gripper 0..100). Widen to include the start
    pose so a rest pose slightly past the range doesn't get yanked inward."""
    limits = {}
    for k, v in start.items():
        lo, hi = (0.0, 100.0) if k == "gripper.pos" else (-100.0, 100.0)
        limits[k] = (min(lo, v), max(hi, v))
    return limits


def status(cmd, speed_name, torque_on, limited):
    joints = " ".join(
        f"{SHORT[k]} {v:.0f}{'!' if k in limited else ''}" for k, v in cmd.items()
    )
    mode = " LIMP" if not torque_on else ""
    line = f"{joints} |{speed_name}{mode}"
    width = shutil.get_terminal_size((80, 20)).columns - 1
    # \x1b[2K clears the line; truncate so it never wraps and floods the terminal
    print(f"\r\x1b[2K{line[:width]}", end="", flush=True)


def main():
    robot = SO101Follower(SO101FollowerConfig(port=PORT, id=ROBOT_ID))
    robot.connect()
    robot.bus.enable_torque()

    start = read_pose(robot)
    limits = make_limits(start)
    target = dict(start)   # where the keys say to go
    cmd = dict(start)      # what we actually send (rate-limited toward target)
    speed_name, step, max_speed = SPEEDS["1"]
    torque_on = True
    quitting = False
    limited = set()

    os.system("")  # enable ANSI escape codes in the Windows console
    signal.signal(signal.SIGINT, on_sigint)
    print(__doc__)
    print("Ready. Starting at slow speed. Ctrl+C = go home and quit (twice = quit now).\n")
    dt = 1.0 / HZ
    next_t = time.perf_counter()
    last_draw = 0.0

    try:
        while True:
            # --- Ctrl+C ---
            if interrupts >= 2:
                break
            keys = []
            if interrupts == 1 and not quitting:
                keys.append("\x1b")   # handle exactly like Esc below

            # --- keyboard ---
            while msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in ("\x00", "\xe0"):      # arrow/function key prefix: discard
                    msvcrt.getwch()
                    continue
                keys.append(ch)

            for ch in keys:
                key = ch.lower()
                if key in KEYMAP and torque_on and not quitting:
                    joint, sign = KEYMAP[key]
                    lo, hi = limits[joint]
                    new = target[joint] + sign * step
                    target[joint] = min(hi, max(lo, new))
                    if new != target[joint]:
                        limited.add(joint)
                    else:
                        limited.discard(joint)
                elif key in SPEEDS:
                    speed_name, step, max_speed = SPEEDS[key]
                elif key == " ":
                    target = dict(cmd)
                elif key == "z" and torque_on:
                    target = dict(start)
                elif key == "p":
                    print("\n" + repr({k: round(v, 1) for k, v in cmd.items()}))
                elif key == "x":
                    if torque_on:
                        robot.bus.disable_torque()
                        torque_on = False
                        print("\nTorque OFF - support the arm. Press x again to lock it in place.")
                    else:
                        pose = read_pose(robot)
                        target, cmd = dict(pose), dict(pose)
                        robot.send_action(cmd)      # goal = here, so it doesn't jump
                        robot.bus.enable_torque()
                        torque_on = True
                        print("\nTorque ON.")
                elif ch in ("\x1b", "\x03"):        # Esc or Ctrl+C
                    if not torque_on:
                        pose = read_pose(robot)
                        target, cmd = dict(pose), dict(pose)
                        robot.send_action(cmd)
                        robot.bus.enable_torque()
                        torque_on = True
                    print("\nReturning to start pose...")
                    target = dict(start)
                    quitting = True

            # --- move cmd toward target, capped at max_speed ---
            if torque_on:
                max_step = max_speed * dt
                for k in cmd:
                    delta = target[k] - cmd[k]
                    cmd[k] += max(-max_step, min(max_step, delta))
                robot.send_action(cmd)

            if quitting and all(abs(target[k] - cmd[k]) < 0.05 for k in cmd):
                break

            now = time.perf_counter()
            if now - last_draw > 0.1:
                status(cmd, speed_name, torque_on, limited)
                last_draw = now

            next_t += dt
            sleep_for = next_t - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_t = time.perf_counter()
    finally:
        print("\nDisconnecting - torque turns off; the arm should be in its rest pose.")
        robot.disconnect()


if __name__ == "__main__":
    main()
