"""Safety filter for commands sent to the real arm (pure numpy, unit-tested)."""

from __future__ import annotations

import numpy as np


class SafetyFilter:
    """Clamp joint targets to a safe box and limit the per-step change.

    All quantities are in the sim frame [rad]. The safe box is the intersection of
    the sim training limits and the arm's calibrated range shrunk by `margin`.
    """

    def __init__(self, sim_low, sim_high, calib_limits, margin: float, max_step: float):
        sim_low, sim_high = np.asarray(sim_low, float), np.asarray(sim_high, float)
        calib_limits = np.asarray(calib_limits, float)
        self.low = np.maximum(sim_low, calib_limits[:, 0] + margin)
        self.high = np.minimum(sim_high, calib_limits[:, 1] - margin)
        if np.any(self.low >= self.high):
            bad = np.nonzero(self.low >= self.high)[0].tolist()
            raise ValueError(f"empty safe range for joint indices {bad}")
        if max_step <= 0:
            raise ValueError("max_step must be positive")
        self.max_step = float(max_step)

    def __call__(self, q_cmd: np.ndarray, q_prev_cmd: np.ndarray) -> np.ndarray:
        """Return the command actually allowed, given the previous command."""
        q = np.clip(np.asarray(q_cmd, float), self.low, self.high)
        step = np.clip(q - q_prev_cmd, -self.max_step, self.max_step)
        return np.clip(q_prev_cmd + step, self.low, self.high)

    def contains(self, q: np.ndarray, tol: float = 1e-9) -> bool:
        q = np.asarray(q, float)
        return bool(np.all(q >= self.low - tol) and np.all(q <= self.high + tol))


def interpolate(q_from: np.ndarray, q_to: np.ndarray, max_step: float) -> list[np.ndarray]:
    """Joint-space waypoints from q_from to q_to (excluding q_from), each step <= max_step."""
    q_from, q_to = np.asarray(q_from, float), np.asarray(q_to, float)
    n = max(1, int(np.ceil(np.max(np.abs(q_to - q_from)) / max_step)))
    return [q_from + (q_to - q_from) * (k / n) for k in range(1, n + 1)]
