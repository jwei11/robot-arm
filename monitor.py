"""Live motor dashboard for the SO-101 follower + bump-to-stop test.

    python monitor.py            # torque ON, arm holds its pose: push it gently and watch
    python monitor.py --limp     # torque OFF: move it by hand (support it!)
    python monitor.py --log run.csv   # also save every reading to a CSV

Columns:
    pos    joint position (degrees from the middle of its calibrated range)
    err    how far the joint is from where it was told to be (deg)
    load   motor effort, % of max torque (+/- = direction)
    mA     motor current        temp  motor temperature (C)       V  supply voltage

With torque ON, pushing a joint makes err AND load rise together: that's a bump.
The dashboard shows BUMP, and with --freeze the arm stops fighting you
(goal is reset to wherever you pushed it).

q or Esc quits. Peak values are shown at the end.
"""

import argparse
import csv
import msvcrt
import os
import time

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from arm_guard import JOINTS, SHORT, TICKS_PER_DEG, BumpGuard, read_telemetry

PORT = "COM20"
ROBOT_ID = "my_follower"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limp", action="store_true", help="torque off so you can move the arm by hand")
    ap.add_argument("--freeze", action="store_true", help="on a bump, re-hold the arm where it was pushed to")
    ap.add_argument("--dev", type=float, default=3.5, help="bump: deg beyond normal lag (default 3.5)")
    ap.add_argument("--hz", type=float, default=20.0)
    ap.add_argument("--log", help="CSV file to write readings to")
    args = ap.parse_args()

    os.system("")  # ANSI escapes on Windows
    robot = SO101Follower(SO101FollowerConfig(port=PORT, id=ROBOT_ID))
    robot.connect()
    bus = robot.bus
    guard = BumpGuard(bus, dev_deg=args.dev)
    mids = {j: (c.range_min + c.range_max) / 2 for j, c in robot.calibration.items()}

    peaks = {j: {"err": 0.0, "load": 0.0, "temp": 0} for j in JOINTS}
    writer = None
    logf = None
    if args.log:
        logf = open(args.log, "w", newline="")
        writer = csv.writer(logf)
        writer.writerow(["t", "joint", "pos_deg", "err_deg", "load_pct", "cur_ma", "temp_c", "volt_v"])

    try:
        if args.limp:
            input("Torque will turn OFF. Support the arm, then press Enter...")
            bus.disable_torque()
        else:
            # hold exactly where it is now
            pos = bus.sync_read("Present_Position", normalize=False)
            bus.sync_write("Goal_Position", pos, normalize=False)
            bus.enable_torque()

        print("\x1b[2J", end="")  # clear screen
        t0 = time.perf_counter()
        bumps = 0
        last_bump = ""
        while True:
            if msvcrt.kbhit() and msvcrt.getwch().lower() in ("q", "\x1b"):
                break
            tel = read_telemetry(bus, full=True)
            t = time.perf_counter() - t0

            if not args.limp:
                bump_joint = guard.update(tel, t)
                if bump_joint:
                    bumps += 1
                    last_bump = f"{time.strftime('%H:%M:%S')}  {guard.describe(bump_joint)}"
                    if args.freeze:
                        guard.freeze()
                        last_bump += "  -> re-held at new pose"
                    guard.reset()

            lines = [
                f"SO-101 monitor   {'LIMP (torque off)' if args.limp else 'HOLDING (torque on)'}   "
                f"t={t:6.1f}s   q/Esc = quit",
                "",
                f"{'joint':<7}{'pos':>8}{'err':>7}{'dev':>7}{'load':>7}{'mA':>7}{'temp':>6}{'V':>6}",
            ]
            for j in JOINTS:
                d = tel[j]
                pos_deg = (d["pos"] - mids[j]) / TICKS_PER_DEG
                p = peaks[j]
                p["err"] = max(p["err"], abs(d["err_deg"]))
                p["load"] = max(p["load"], abs(d["load_pct"]))
                p["temp"] = max(p["temp"], d["temp_c"])
                flag = ""
                dev = guard.dev.get(j, 0.0)
                if not args.limp and j != "gripper" and abs(dev) > args.dev:
                    flag = "  << BUMP"
                if d["temp_c"] >= 55:
                    flag += "  HOT"
                lines.append(
                    f"{SHORT[j]:<7}{pos_deg:8.1f}{d['err_deg']:7.1f}{dev:7.1f}{d['load_pct']:7.0f}"
                    f"{d['cur_ma']:7.0f}{d['temp_c']:6d}{d['volt_v']:6.1f}{flag}"
                )
                if writer:
                    writer.writerow(
                        [
                            f"{t:.3f}",
                            j,
                            f"{pos_deg:.2f}",
                            f"{d['err_deg']:.2f}",
                            d["load_pct"],
                            f"{d['cur_ma']:.0f}",
                            d["temp_c"],
                            d["volt_v"],
                        ]
                    )
            lines += [
                "",
                f"bump rule: dev (error beyond normal lag) > {args.dev} deg   bumps: {bumps}",
                f"last: {last_bump}",
            ]
            print("\x1b[H" + "\n".join(line + "\x1b[K" for line in lines), end="", flush=True)
            time.sleep(1 / args.hz)
    finally:
        print("\n\nPeaks:")
        for j in JOINTS:
            p = peaks[j]
            print(f"  {SHORT[j]:<7} err {p['err']:5.1f} deg   load {p['load']:5.0f}%   temp {p['temp']} C")
        if logf:
            logf.close()
            print(f"Log saved to {args.log}")
        if not args.limp:
            print("Disconnecting - torque turns off. Support the arm or leave it resting.")
        robot.disconnect()


if __name__ == "__main__":
    main()
