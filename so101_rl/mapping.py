"""Mapping between MuJoCo joint angles [rad] and LeRobot 0.4.4 joint values.

No LeRobot import is needed: the conversions below mirror LeRobot 0.4.4's
`MotorsBus._normalize` / `_unnormalize` (lerobot/motors/motors_bus.py) exactly,
and are checked against the real LeRobot code in tests/test_mapping.py when
LeRobot is installed.

How LeRobot 0.4.4 reports SO-101 arm joints
-------------------------------------------
`SO101FollowerConfig.use_degrees` defaults to **True**. Then for the 5 arm joints:

    deg = (raw - mid) * 360 / 4095,   mid = (range_min + range_max) / 2

where raw is the servo's Present_Position tick (after the homing offset written at
calibration) and range_min/max are the calibrated ticks from
~/.cache/huggingface/lerobot/calibration/robots/so_follower/<id>.json.
Note: degrees mode ignores `drive_mode` and does NOT clamp on write.

With `use_degrees=False` the arm joints are RANGE_M100_100:

    value = (clip(raw, min, max) - min) / (max - min) * 200 - 100   (negated if drive_mode)

The gripper is always RANGE_0_100 and is not used by the policy.

Sim <-> real convention (default)
---------------------------------
The official MJCF `so101_new_calib.xml` puts each joint's zero at the middle of its
range, and LeRobot's own SO-101 kinematics (lerobot/model/kinematics.py,
`RobotKinematics.forward_kinematics`) feeds degrees-mode values straight into the
same model with `np.deg2rad`, with no sign flips or offsets. So by default:

    q_mujoco [rad] = sign * deg2rad(lerobot_deg - offset_deg),  sign=+1, offset_deg=0

`signs` and `offsets_deg` are exposed so they can be corrected per joint if a
hardware check (run_real.py --check-mapping) shows a mismatch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from so101_rl.kinematics import ARM_JOINTS

STS3215_MAX_RES = 4095  # LeRobot uses model_resolution - 1 for the degree conversion
DEFAULT_CALIBRATION = (
    Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration" / "robots" / "so_follower" / "my_follower.json"
)


@dataclass(frozen=True)
class MotorCalib:
    id: int
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int


def load_calibration(path: str | Path = DEFAULT_CALIBRATION) -> dict[str, MotorCalib]:
    with open(path) as f:
        raw = json.load(f)
    return {name: MotorCalib(**vals) for name, vals in raw.items()}


# ---------------------------------------------------------------- LeRobot math
def lerobot_normalize(raw: float, cal: MotorCalib, use_degrees: bool) -> float:
    """raw servo ticks -> LeRobot value (mirrors MotorsBus._normalize for arm joints)."""
    lo, hi = cal.range_min, cal.range_max
    if use_degrees:
        mid = (lo + hi) / 2
        return (raw - mid) * 360 / STS3215_MAX_RES
    bounded = min(hi, max(lo, raw))
    norm = ((bounded - lo) / (hi - lo)) * 200 - 100
    return -norm if cal.drive_mode else norm


def lerobot_unnormalize(value: float, cal: MotorCalib, use_degrees: bool, truncate: bool = False) -> float:
    """LeRobot value -> raw servo ticks (mirrors MotorsBus._unnormalize for arm joints).
    LeRobot truncates with int(); pass truncate=True to reproduce that exactly."""
    lo, hi = cal.range_min, cal.range_max
    if use_degrees:
        mid = (lo + hi) / 2
        raw = value * STS3215_MAX_RES / 360 + mid
    else:
        v = -value if cal.drive_mode else value
        v = min(100.0, max(-100.0, v))
        raw = ((v + 100) / 200) * (hi - lo) + lo
    return float(int(raw)) if truncate else raw


# ---------------------------------------------------------------- mapping
class SimRealMapping:
    """Converts between 5 MuJoCo arm angles [rad] and LeRobot action/observation dicts."""

    def __init__(
        self,
        calibration: dict[str, MotorCalib],
        use_degrees: bool = True,
        signs: dict[str, int] | None = None,
        offsets_deg: dict[str, float] | None = None,
    ):
        missing = set(ARM_JOINTS) - set(calibration)
        if missing:
            raise ValueError(f"calibration is missing joints: {sorted(missing)}")
        self.cal = calibration
        self.use_degrees = use_degrees
        self.signs = {j: 1 for j in ARM_JOINTS} | (signs or {})
        self.offsets_deg = {j: 0.0 for j in ARM_JOINTS} | (offsets_deg or {})
        for j, s in self.signs.items():
            if s not in (1, -1):
                raise ValueError(f"sign for {j} must be +1 or -1, got {s}")

    # Per joint: LeRobot value <-> "LeRobot degrees" (deg mode is the reference frame).
    def _value_to_deg(self, j: str, value: float) -> float:
        if self.use_degrees:
            return value
        return lerobot_normalize(lerobot_unnormalize(value, self.cal[j], False), self.cal[j], True)

    def _deg_to_value(self, j: str, deg: float) -> float:
        if self.use_degrees:
            return deg
        return lerobot_normalize(lerobot_unnormalize(deg, self.cal[j], True), self.cal[j], False)

    def real_to_sim(self, obs: dict[str, float]) -> np.ndarray:
        """LeRobot observation/action dict ('<joint>.pos' keys) -> q [rad] (5,)."""
        q = np.empty(len(ARM_JOINTS))
        for i, j in enumerate(ARM_JOINTS):
            deg = self._value_to_deg(j, float(obs[f"{j}.pos"]))
            q[i] = self.signs[j] * np.radians(deg - self.offsets_deg[j])
        return q

    def sim_to_real(self, q: np.ndarray) -> dict[str, float]:
        """q [rad] (5,) -> LeRobot action dict ('<joint>.pos' keys), arm joints only."""
        out = {}
        for i, j in enumerate(ARM_JOINTS):
            deg = self.signs[j] * np.degrees(float(q[i])) + self.offsets_deg[j]
            out[f"{j}.pos"] = float(self._deg_to_value(j, deg))
        return out

    def calibrated_limits_rad(self) -> np.ndarray:
        """(5, 2) sim-frame [low, high] corresponding to each joint's calibrated tick range."""
        lim = np.empty((len(ARM_JOINTS), 2))
        for i, j in enumerate(ARM_JOINTS):
            c = self.cal[j]
            a = self.signs[j] * np.radians(lerobot_normalize(c.range_min, c, True) - self.offsets_deg[j])
            b = self.signs[j] * np.radians(lerobot_normalize(c.range_max, c, True) - self.offsets_deg[j])
            lim[i] = (min(a, b), max(a, b))
        return lim
