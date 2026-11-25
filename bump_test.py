"""Measure tracking error and load while a joint moves: free vs. blocked by your hand.

    python bump_test.py                       # wrist_roll, +-30 deg
    python bump_test.py --joint shoulder_pan --amp 20

Phase 1 (free):    the joint swings back and forth on its own.
Phase 2 (blocked): same swing; grab the arm and stop it mid-swing for ~1 s, a few times.
Writes bump_test.csv and prints a summary so thresholds can be picked from real data.
"""

import argparse
import csv
import math
import time

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from arm_guard import TICKS_PER_DEG, read_telemetry

PORT = "COM20"
ROBOT_ID = "my_follower"
HZ = 50


def swing(bus, joint, center, amp_ticks, speed_dps, cycles, phase, writer):
    """Triangle-ish (cosine) swing around center; logs every tick."""
    period = 4 * (amp_ticks / TICKS_PER_DEG) / speed_dps  # seconds for one full back-and-forth
    t0 = time.perf_counter()
    rows = []
    n = int(period * cycles * HZ)
    for i in range(n):
        t = i / HZ
        goal = int(center + amp_ticks * math.sin(2 * math.pi * t / period))
        bus.sync_write("Goal_Position", {joint: goal}, normalize=False)
        tel = read_telemetry(bus, full=False)[joint]
        row = [phase, f"{t:.3f}", goal, tel["pos"], f"{tel['err_deg']:.2f}", tel["load_pct"]]
        rows.append(row)
        writer.writerow(row)
        if i % 10 == 0:
            print(f"\r  {phase:8s} t={t:5.1f}s  err {tel['err_deg']:+6.1f} deg  load {tel['load_pct']:+6.1f}%   ",
                  end="", flush=True)
        wait = t0 + (i + 1) / HZ - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
    # ease back to center
    bus.sync_write("Goal_Position", {joint: int(center)}, normalize=False)
    time.sleep(0.8)
    print()
    return rows


def summary(rows, label):
    errs = sorted(abs(float(r[4])) for r in rows)
    loads = sorted(abs(float(r[5])) for r in rows)
    p = lambda a, q: a[int(q * (len(a) - 1))]  # noqa: E731
    print(f"  {label:8s} err  p50 {p(errs, .5):5.1f}  p95 {p(errs, .95):5.1f}  max {errs[-1]:5.1f} deg   |   "
          f"load p50 {p(loads, .5):5.1f}  p95 {p(loads, .95):5.1f}  max {loads[-1]:5.1f} %")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--joint", default="wrist_roll",
                    choices=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"])
    ap.add_argument("--amp", type=float, default=30.0, help="swing amplitude, deg (max 40)")
    ap.add_argument("--speed", type=float, default=45.0, help="average speed, deg/s")
    ap.add_argument("--cycles", type=int, default=3)
    args = ap.parse_args()
    amp = min(args.amp, 40.0)

    robot = SO101Follower(SO101FollowerConfig(port=PORT, id=ROBOT_ID))
    robot.connect()
    bus = robot.bus
    cal = robot.calibration[args.joint]
    try:
        pos = bus.sync_read("Present_Position", normalize=False)
        bus.sync_write("Goal_Position", pos, normalize=False)  # hold where it is
        bus.enable_torque()
        center = pos[args.joint]
        amp_ticks = amp * TICKS_PER_DEG
        lo, hi = cal.range_min + 3 * TICKS_PER_DEG, cal.range_max - 3 * TICKS_PER_DEG
        if cal.range_max - cal.range_min < 4000:  # joint has real limits: keep the swing inside them
            center = min(max(center, lo + amp_ticks), hi - amp_ticks)
            if center != pos[args.joint]:
                print(f"Note: {args.joint} is near its limit; the swing is shifted inward.")
        print(f"{args.joint}: swinging +-{amp:.0f} deg at ~{args.speed:.0f} deg/s, {args.cycles} cycles.")
        input("Clear the space around the arm. Phase 1 (FREE, don't touch). Press Enter...")

        with open("bump_test.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["phase", "t", "goal", "pos", "err_deg", "load_pct"])
            free = swing(bus, args.joint, center, amp_ticks, args.speed, args.cycles, "free", w)
            input("\nPhase 2 (BLOCKED): grab the arm and stop the swing for ~1 s, a few times. "
                  "Let go if it feels strong. Press Enter to start...")
            blocked = swing(bus, args.joint, center, amp_ticks, args.speed, args.cycles + 1, "blocked", w)

        print("\nSummary:")
        summary(free, "free")
        summary(blocked, "blocked")
        print("Saved bump_test.csv")
    finally:
        print("Disconnecting - torque turns off. Support the arm.")
        robot.disconnect()


if __name__ == "__main__":
    main()
