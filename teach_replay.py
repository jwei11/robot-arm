"""Teach & replay for the SO-101 follower.

Teach:  torque goes off, you pose the arm by hand and press Enter at each waypoint.
Replay: torque comes on and the arm moves smoothly through the waypoints
        (speed-limited, eased start/stop), then returns to where it started.

Usage:
    python teach_replay.py                      # teach, save, then replay
    python teach_replay.py --replay-only        # replay saved waypoints.json
    python teach_replay.py --speed 40 --loops 3 # slower, repeat 3 times
"""

import argparse
import json
import math
import time
from pathlib import Path

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

PORT = "COM20"
ROBOT_ID = "my_follower"
HZ = 50                 # control loop rate
MIN_SEGMENT_S = 0.6     # never move between waypoints faster than this


def read_pose(robot):
    obs = robot.get_observation()
    return {k: float(v) for k, v in obs.items() if k.endswith(".pos")}


def fmt(pose):
    return "  ".join(f"{k.removesuffix('.pos')}={v:7.1f}" for k, v in pose.items())


def move_smooth(robot, start, goal, speed):
    """Interpolate start -> goal with a cosine ease, limited to `speed` units/s on the fastest joint."""
    biggest = max(abs(goal[k] - start[k]) for k in goal)
    duration = max(MIN_SEGMENT_S, biggest / speed)
    steps = max(1, int(duration * HZ))
    t0 = time.perf_counter()
    for i in range(1, steps + 1):
        s = 0.5 - 0.5 * math.cos(math.pi * i / steps)  # 0 -> 1, zero velocity at both ends
        robot.send_action({k: start[k] + (goal[k] - start[k]) * s for k in goal})
        # keep a steady rate
        sleep_for = t0 + i / HZ - time.perf_counter()
        if sleep_for > 0:
            time.sleep(sleep_for)


def teach(robot):
    print("\n=== TEACH MODE ===")
    print("Torque is turning OFF. Support the arm with your hand so it doesn't drop.")
    input("Hold the arm, then press Enter...")
    robot.bus.disable_torque()

    waypoints = []
    print("\nPose the arm, then press Enter to record a waypoint.")
    print("Type 'u' + Enter to undo the last one, 'd' + Enter when done.\n")
    while True:
        cmd = input(f"[{len(waypoints)} recorded] Enter=record  u=undo  d=done > ").strip().lower()
        if cmd == "d":
            break
        if cmd == "u":
            if waypoints:
                waypoints.pop()
                print("  removed last waypoint")
            continue
        pose = read_pose(robot)
        waypoints.append(pose)
        print(f"  #{len(waypoints)}: {fmt(pose)}")

    # Before torque comes back, set the goal to where the arm is NOW so it doesn't jump.
    print("\nHold the arm in a safe pose; torque is coming back ON.")
    input("Press Enter to re-enable torque...")
    robot.send_action(read_pose(robot))
    robot.bus.enable_torque()
    return waypoints


def replay(robot, waypoints, speed, loops, hold):
    home = read_pose(robot)
    print(f"\n=== REPLAY: {len(waypoints)} waypoints x {loops} loop(s), speed {speed}/s ===")
    current = home
    for loop in range(loops):
        for i, wp in enumerate(waypoints, 1):
            print(f"  loop {loop + 1}  -> waypoint {i}")
            move_smooth(robot, current, wp, speed)
            current = wp
            time.sleep(hold)
    print("  returning to start pose")
    move_smooth(robot, current, home, speed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="waypoints.json")
    ap.add_argument("--replay-only", action="store_true")
    ap.add_argument("--speed", type=float, default=60.0, help="max joint speed, units/s (default 60)")
    ap.add_argument("--loops", type=int, default=1)
    ap.add_argument("--hold", type=float, default=0.3, help="pause at each waypoint, seconds")
    args = ap.parse_args()
    path = Path(args.file)

    robot = SO101Follower(SO101FollowerConfig(port=PORT, id=ROBOT_ID))
    robot.connect()
    try:
        robot.bus.enable_torque()
        if args.replay_only:
            waypoints = json.loads(path.read_text())
            print(f"Loaded {len(waypoints)} waypoints from {path}")
        else:
            waypoints = teach(robot)
            if not waypoints:
                print("No waypoints recorded.")
                return
            path.write_text(json.dumps(waypoints, indent=2))
            print(f"Saved {len(waypoints)} waypoints to {path}")

        while True:
            replay(robot, waypoints, args.speed, args.loops, args.hold)
            again = input("\nReplay again? Enter=yes, q=quit > ").strip().lower()
            if again == "q":
                break
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        print("Disconnecting - torque will turn off, so support the arm or leave it in a rest pose.")
        robot.disconnect()


if __name__ == "__main__":
    main()
