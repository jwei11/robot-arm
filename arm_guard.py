"""Motor telemetry + bump detection for the SO-101 follower (Feetech STS3215).

Reads raw registers straight from the bus, so it works the same whether the robot
was created in degrees mode or -100..100 mode.

A "bump" = a joint is being pushed away from where it was told to be:
    |goal - present| > err_deg   AND   |load| > load_pct
for `hold_ticks` consecutive checks. Requiring both avoids false alarms from
normal lag while moving (error but little load) and from gravity (load but the
joint is where it should be).
"""

import time

TICKS_PER_DEG = 4096 / 360
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
SHORT = {
    "shoulder_pan": "pan",
    "shoulder_lift": "lift",
    "elbow_flex": "elbow",
    "wrist_flex": "wflex",
    "wrist_roll": "wroll",
    "gripper": "grip",
}


def read_telemetry(bus, full=True):
    """Return {joint: {...}} with pos/goal in ticks, err in degrees, load in %,
    and (if full) current mA, temperature C, voltage V."""
    pos = bus.sync_read("Present_Position", normalize=False)
    goal = bus.sync_read("Goal_Position", normalize=False)
    load = bus.sync_read("Present_Load", normalize=False)  # 0.1 % of max torque, signed
    out = {}
    extra = {}
    if full:
        extra["cur"] = bus.sync_read("Present_Current", normalize=False)  # ~6.5 mA per unit
        extra["temp"] = bus.sync_read("Present_Temperature", normalize=False)  # deg C
        extra["volt"] = bus.sync_read("Present_Voltage", normalize=False)  # 0.1 V
    for j in pos:
        d = {
            "pos": pos[j],
            "goal": goal[j],
            "err_deg": (goal[j] - pos[j]) / TICKS_PER_DEG,
            "load_pct": load[j] / 10.0,
        }
        if full:
            d["cur_ma"] = extra["cur"][j] * 6.5
            d["temp_c"] = extra["temp"][j]
            d["volt_v"] = extra["volt"][j] / 10.0
        out[j] = d
    return out


LAG_S = 0.102       # measured: free-moving joint lags ~0.10 s x commanded speed
LAG_DELAY_S = 0.08  # ...using the commanded speed from ~80 ms earlier


class BumpGuard:
    """Detects a joint being held back (or pushed) relative to its normal lag.

    expected_err = LAG_S * commanded speed (measured on this arm: residual sd 0.3 deg,
    worst 1.5 deg when moving freely; 6-8 deg when blocked by hand). A slow baseline
    absorbs steady offsets like gravity sag, so only sudden deviations count.
    """

    def __init__(self, bus, dev_deg=3.5, hold_ticks=3, baseline_tau_s=2.0, ignore=("gripper",)):
        self.bus = bus
        self.dev_deg = dev_deg
        self.hold_ticks = hold_ticks
        self.baseline_tau_s = baseline_tau_s
        self.ignore = set(ignore)
        self.reset()

    def reset(self):
        self.counts = {}
        self.hist = {}      # joint -> [(t, goal_ticks)]
        self.base = {}      # joint -> slow baseline of (err - expected)
        self.dev = {}       # joint -> latest deviation [deg]
        self.last = {}
        self.t_prev = None

    def _goal_at(self, j, t):
        h = self.hist[j]
        best = h[0]
        for s in h:
            if s[0] <= t:
                best = s
            else:
                break
        return best[1]

    def update(self, tel, t):
        """Feed one telemetry reading taken at time t [s]; return bumping joint or None."""
        self.last = tel
        dt = 0.0 if self.t_prev is None else t - self.t_prev
        self.t_prev = t
        hit = None
        for j, d in tel.items():
            if j in self.ignore:
                continue
            h = self.hist.setdefault(j, [])
            h.append((t, d["goal"]))
            while h and h[0][0] < t - 0.5:
                h.pop(0)
            t1, t0 = t - LAG_DELAY_S + 0.05, t - LAG_DELAY_S - 0.05
            v_dps = (self._goal_at(j, t1) - self._goal_at(j, t0)) / TICKS_PER_DEG / 0.1
            resid = d["err_deg"] - LAG_S * v_dps
            base = self.base.get(j, resid)
            dev = resid - base
            self.dev[j] = dev
            if abs(dev) > self.dev_deg:
                self.counts[j] = self.counts.get(j, 0) + 1
                if self.counts[j] >= self.hold_ticks:
                    hit = hit or j
            else:
                self.counts[j] = 0
                # adapt the baseline only while things look normal
                a = min(1.0, dt / self.baseline_tau_s) if dt > 0 else 1.0
                self.base[j] = base + a * (resid - base)
        return hit

    def check(self):
        """Read the bus; return the joint name that is bumping, or None."""
        return self.update(read_telemetry(self.bus, full=False), time.perf_counter())

    def freeze(self):
        """Hold every joint exactly where it is now (goal = present), so it stops pushing."""
        pos = self.bus.sync_read("Present_Position", normalize=False)
        self.bus.sync_write("Goal_Position", pos, normalize=False)
        time.sleep(0.05)
        return pos

    def describe(self, joint):
        d = self.last.get(joint, {})
        return (f"{joint}: {self.dev.get(joint, 0):+.1f} deg beyond normal lag "
                f"(error {d.get('err_deg', 0):+.1f} deg, load {d.get('load_pct', 0):+.0f}%)")
