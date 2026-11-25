"""Record a hand-guided motion at 50 Hz and play it back with its original timing.

    python record_motion.py record wave          # teach a motion called "wave"
    python record_motion.py play wave            # play it back
    python record_motion.py play wave --rate 0.5 --loops 3   # half speed, 3 times
    python record_motion.py list                 # show saved motions

Motions are saved in ./motions/<name>.json.

Recording: torque goes off, you move the arm by hand, Enter starts and Enter stops.
Playback:  the arm eases to the first frame, replays every frame on the original
           timeline (scaled by --rate), then eases back to where it started.
           Esc or Ctrl+C during playback stops it and eases back to the start.
"""

import argparse
import json
import math
import msvcrt
import signal
import time
from pathlib import Path

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

PORT = "COM20"
ROBOT_ID = "my_follower"
HZ = 50
MOTION_DIR = Path("motions")

stop_requested = False


def on_sigint(signum, frame):
    global stop_requested
    stop_requested = True


def esc_pressed():
    hit = False
    while msvcrt.kbhit():
        if msvcrt.getwch() == "\x1b":
            hit = True
    return hit


def read_pose(robot):
    return {k: float(v) for k, v in robot.get_observation().items() if k.endswith(".pos")}


def lock_here(robot):
    """Turn torque on without a jump: set the goal to the current position first."""
    robot.send_action(read_pose(robot))
    robot.bus.enable_torque()


def move_smooth(robot, start, goal, speed=60.0, min_s=0.6):
    biggest = max(abs(goal[k] - start[k]) for k in goal)
    steps = max(1, int(max(min_s, biggest / speed) * HZ))
    t0 = time.perf_counter()
    for i in range(1, steps + 1):
        s = 0.5 - 0.5 * math.cos(math.pi * i / steps)
        robot.send_action({k: start[k] + (goal[k] - start[k]) * s for k in goal})
        wait = t0 + i / HZ - time.perf_counter()
        if wait > 0:
            time.sleep(wait)


def smooth_frames(frames, window):
    """Centered moving average over each joint to remove hand jitter."""
    if window <= 1:
        return frames
    half = window // 2
    out = []
    for i in range(len(frames)):
        lo, hi = max(0, i - half), min(len(frames), i + half + 1)
        chunk = frames[lo:hi]
        out.append({k: sum(f[k] for f in chunk) / len(chunk) for k in frames[i]})
    return out


# ---------------------------------------------------------------- record
def record(robot, name):
    path = MOTION_DIR / f"{name}.json"
    if path.exists() and input(f"{path} exists. Overwrite? y/N > ").strip().lower() != "y":
        return None

    home = read_pose(robot)
    print("\nTorque is turning OFF. Hold the arm so it doesn't drop.")
    input("Holding it? Press Enter...")
    robot.bus.disable_torque()

    input("Move the arm to where the motion should START, then press Enter to begin recording...")
    while msvcrt.kbhit():
        msvcrt.getwch()
    print("RECORDING - move the arm. Press Enter to stop.")

    frames = []
    t0 = time.perf_counter()
    next_t = t0
    while True:
        if msvcrt.kbhit() and msvcrt.getwch() in ("\r", "\n"):
            break
        if stop_requested:
            break
        frames.append(read_pose(robot))
        if len(frames) % 10 == 0:
            print(f"\r  {len(frames) / HZ:5.1f} s   {len(frames)} frames", end="", flush=True)
        next_t += 1 / HZ
        wait = next_t - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        else:
            next_t = time.perf_counter()
    print(f"\nStopped: {len(frames)} frames, {len(frames) / HZ:.1f} s")

    print("\nTorque is coming back ON where the arm is now.")
    input("Hold the arm in a safe pose and press Enter...")
    lock_here(robot)

    if len(frames) < 5:
        print("Too short, not saved.")
        return None
    MOTION_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps({"hz": HZ, "home": home, "frames": frames}))
    print(f"Saved to {path}")
    return path


# ---------------------------------------------------------------- play
def play(robot, path, rate, loops, smooth, vmax):
    global stop_requested
    data = json.loads(path.read_text())
    hz = data["hz"]
    frames = smooth_frames(data["frames"], smooth)
    duration = (len(frames) - 1) / hz
    print(f"\nPlaying {path.name}: {duration:.1f} s x {loops} at {rate}x speed (Esc / Ctrl+C to stop)")

    home = read_pose(robot)
    move_smooth(robot, home, frames[0])
    cmd = dict(frames[0])
    max_step = vmax / HZ
    clipped = 0

    for loop in range(loops):
        if loop:
            move_smooth(robot, cmd, frames[0])
            cmd = dict(frames[0])
        print(f"  loop {loop + 1}/{loops}")
        t0 = time.perf_counter()
        i = 0
        while True:
            if stop_requested or esc_pressed():
                print("  stopped")
                stop_requested = False
                move_smooth(robot, cmd, home)
                return
            i += 1
            t = (i / HZ) * rate  # position on the recording's timeline
            if t >= duration:
                break
            # linear interpolation between recorded frames
            f = t * hz
            a = int(f)
            frac = f - a
            fa, fb = frames[a], frames[min(a + 1, len(frames) - 1)]
            target = {k: fa[k] + (fb[k] - fa[k]) * frac for k in fa}
            # safety: cap joint speed even if the recording was fast
            for k in cmd:
                d = target[k] - cmd[k]
                if abs(d) > max_step:
                    clipped += 1
                cmd[k] += max(-max_step, min(max_step, d))
            robot.send_action(cmd)
            wait = t0 + i / HZ - time.perf_counter()
            if wait > 0:
                time.sleep(wait)

    if clipped:
        print(
            f"  note: speed cap ({vmax:.0f}/s) slowed {clipped} steps - use --vmax higher or --rate lower if it lagged"
        )
    print("  returning to start pose")
    move_smooth(robot, cmd, home)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["record", "play", "list"])
    ap.add_argument("name", nargs="?")
    ap.add_argument("--rate", type=float, default=1.0, help="playback speed multiplier (0.5 = half speed)")
    ap.add_argument("--loops", type=int, default=1)
    ap.add_argument("--smooth", type=int, default=5, help="smoothing window in frames (1 = off)")
    ap.add_argument("--vmax", type=float, default=150.0, help="max joint speed during playback, units/s")
    args = ap.parse_args()

    if args.mode == "list":
        files = sorted(MOTION_DIR.glob("*.json")) if MOTION_DIR.exists() else []
        for p in files:
            d = json.loads(p.read_text())
            print(f"  {p.stem:<20} {(len(d['frames']) - 1) / d['hz']:5.1f} s")
        if not files:
            print("No motions yet. Record one: python record_motion.py record <name>")
        return
    if not args.name:
        ap.error("give the motion a name, e.g. record wave")

    signal.signal(signal.SIGINT, on_sigint)
    robot = SO101Follower(SO101FollowerConfig(port=PORT, id=ROBOT_ID))
    robot.connect()
    try:
        robot.bus.enable_torque()
        if args.mode == "record":
            path = record(robot, args.name)
            if path is None:
                return
        else:
            path = MOTION_DIR / f"{args.name}.json"
            if not path.exists():
                print(f"No motion named {args.name!r}. Try: python record_motion.py list")
                return
        while True:
            if input("\nPlay it? Enter=play, q=quit > ").strip().lower() == "q":
                break
            play(robot, path, args.rate, args.loops, args.smooth, args.vmax)
    finally:
        print("Disconnecting - torque turns off; the arm should be in its rest pose.")
        robot.disconnect()


if __name__ == "__main__":
    main()
