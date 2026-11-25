"""Reachable target sampling: random joint configs -> forward kinematics -> filter."""

from __future__ import annotations

import numpy as np

from so101_rl.kinematics import Kinematics


def sample_reachable_target(
    kin: Kinematics,
    rng: np.random.Generator,
    low: np.ndarray,
    high: np.ndarray,
    min_z: float,
    min_radius: float,
    max_bearing_deg: float | None = None,
    max_tries: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (target_xyz, q) where q is a collision-free config within [low, high]
    whose end-effector is at target_xyz, above min_z and at least min_radius from the
    base's vertical axis, and (if max_bearing_deg is set) within that many degrees of the
    arm's forward (+x) direction. Because the target is the FK of a valid config, it is reachable
    by construction."""
    for _ in range(max_tries):
        q = rng.uniform(low, high)
        p = kin.fk(q)
        if p[2] < min_z or np.hypot(p[0], p[1]) < min_radius:
            continue
        if max_bearing_deg is not None and abs(np.degrees(np.arctan2(p[1], p[0]))) > max_bearing_deg:
            continue
        if kin.in_collision(q):
            continue
        return p, q
    raise RuntimeError("Could not sample a reachable target; check limits / filters.")


class WorkspaceSampler:
    """Samples targets evenly over the reachable workspace instead of evenly over joint space.

    Uniformly random joint angles crowd targets into some regions and leave others almost
    empty (e.g. close-in, high points). This sampler draws a large pool of FK points from
    random joint configs, bins them into ``cell`` sized cubes, and at sample time picks an
    occupied cube uniformly, then a pool point inside it. Every target is still the FK of a
    valid config, so it stays reachable by construction. The pool uses its own fixed seed,
    so every env shares the same pool; only the per-episode choice uses the env's rng.
    """

    def __init__(
        self,
        kin: Kinematics,
        low: np.ndarray,
        high: np.ndarray,
        min_z: float,
        min_radius: float,
        max_bearing_deg: float | None = None,
        cell: float = 0.03,
        n_pool: int = 300_000,
        pool_seed: int = 0,
    ):
        self.kin = kin
        pool_rng = np.random.default_rng(pool_seed)
        qs = pool_rng.uniform(low, high, size=(n_pool, len(low)))
        ps = np.array([kin.fk(q) for q in qs])
        keep = (ps[:, 2] >= min_z) & (np.hypot(ps[:, 0], ps[:, 1]) >= min_radius)
        if max_bearing_deg is not None:
            keep &= np.abs(np.degrees(np.arctan2(ps[:, 1], ps[:, 0]))) <= max_bearing_deg
        self.q, self.p = qs[keep], ps[keep]
        if len(self.p) == 0:
            raise RuntimeError("Workspace pool is empty; check limits / filters.")
        _, cell_index = np.unique(np.floor(self.p / cell).astype(np.int64), axis=0, return_inverse=True)
        cell_index = cell_index.ravel()
        order = np.argsort(cell_index, kind="stable")
        self.cells = np.split(order, np.cumsum(np.bincount(cell_index))[:-1])

    def __call__(self, rng: np.random.Generator, max_tries: int = 1000) -> tuple[np.ndarray, np.ndarray]:
        for _ in range(max_tries):
            members = self.cells[rng.integers(len(self.cells))]
            i = members[rng.integers(len(members))]
            if not self.kin.in_collision(self.q[i]):
                return self.p[i].copy(), self.q[i].copy()
        raise RuntimeError("Could not sample a collision-free target.")
