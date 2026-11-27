"""Environment and domain-randomization configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class DomainRandomization:
    """Per-episode randomization, sampled on every reset. Everything is OFF by default.

    Ranges are (low, high), sampled uniformly. A range of (1, 1) for scales or
    (0, 0) for latency means "no randomization" for that item, so each item can
    be turned on independently even when ``enabled`` is True.
    """

    enabled: bool = False
    # Multiplicative scales on the model's nominal values.
    damping_scale: tuple[float, float] = (1.0, 1.0)
    armature_scale: tuple[float, float] = (1.0, 1.0)
    frictionloss_scale: tuple[float, float] = (1.0, 1.0)
    # Position-actuator stiffness (kp) and damping (kv) are scaled together.
    # Below 1.0 models a weaker servo, e.g. 7.4 V STS3215s run from a 5 V supply.
    actuator_gain_scale: tuple[float, float] = (1.0, 1.0)
    # Gaussian noise std on observed joint positions [rad]. Observed velocity is the
    # finite difference of the noisy positions, exactly like on the real arm.
    obs_noise_std: float = 0.0
    # Number of control steps an action is delayed before it reaches the servos.
    action_latency_steps: tuple[int, int] = (0, 0)

    @classmethod
    def sim2real(cls) -> DomainRandomization:
        """A reasonable starting point for sim-to-real training."""
        return cls(
            enabled=True,
            damping_scale=(0.7, 1.5),
            armature_scale=(0.7, 1.5),
            frictionloss_scale=(0.5, 2.0),
            actuator_gain_scale=(0.4, 1.2),
            obs_noise_std=float(np.radians(0.3)),
            action_latency_steps=(0, 2),
        )


@dataclass
class EnvConfig:
    control_hz: float = 20.0
    episode_steps: int = 100  # 5 s at 20 Hz
    # Max joint-target change per control step [rad] (0.05 rad = 2.9 deg -> 57 deg/s).
    max_delta: float = 0.05
    # Joint targets are kept this far inside the model's joint limits [rad].
    limit_margin: float = float(np.radians(5.0))
    success_radius: float = 0.02  # [m]
    success_bonus: float = 0.5  # per step while within success_radius
    action_penalty: float = 0.01  # times ||a||^2, a in [-1, 1]^5
    # Dense precision term: precision_weight * (1 - tanh(dist / precision_scale)).
    precision_weight: float = 1.0
    precision_scale: float = 0.05  # [m]
    # Arm start pose [rad] (5 joints) plus uniform noise of +/- home_noise.
    home_qpos: tuple[float, ...] = (0.0, -0.9, 0.9, 0.6, 0.0)
    home_noise: float = 0.1
    gripper_q: float = 0.0  # gripper joint is held here [rad]
    # Target sampling (see targets.py).
    target_min_z: float = 0.05  # [m] above the table
    target_min_radius: float = 0.12  # [m] horizontal distance from the base axis
    # Targets are kept within this many degrees of straight ahead (+x); None = all around.
    # Targets behind the arm need an over-the-top pose and were most of the misses.
    target_max_bearing_deg: float | None = 90.0
    # "workspace": even over the reachable volume (targets.WorkspaceSampler, 3 cm cells).
    # "joint": FK of uniformly random joint angles (uneven in space; older runs used this).
    target_sampling: str = "workspace"
    dr: DomainRandomization = field(default_factory=DomainRandomization)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> EnvConfig:
        d = dict(d)
        task = d.pop("task", "reach")
        if task != "reach":
            raise ValueError(f"this is a {task!r} config, not a reach config")
        # Configs saved before the bearing filter existed sampled targets all around.
        d.setdefault("target_max_bearing_deg", None)
        d.setdefault("target_sampling", "joint")
        dr = DomainRandomization(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.pop("dr", {}).items()})
        d = {k: tuple(v) if isinstance(v, list) else v for k, v in d.items()}
        return cls(dr=dr, **d)


@dataclass
class PushConfig:
    """Push a 4 cm cube across the table onto a goal disc (see push_env.py)."""

    task: str = "push"
    control_hz: float = 20.0
    episode_steps: int = 200  # 10 s at 20 Hz
    max_delta: float = 0.05  # [rad] per control step, as in the reach task
    limit_margin: float = float(np.radians(5.0))
    home_qpos: tuple[float, ...] = (0.0, -0.9, 0.9, 0.6, 0.0)
    home_noise: float = 0.1
    gripper_q: float = 0.0
    # Block start: horizontal distance from the base axis and bearing from straight ahead.
    block_radius: tuple[float, float] = (0.18, 0.28)  # [m]
    block_max_bearing_deg: float = 35.0
    block_yaw_range_deg: float = 45.0
    # Goal: this far from the block start, and inside goal_radius / goal_max_bearing_deg.
    goal_offset: tuple[float, float] = (0.06, 0.15)  # [m]
    goal_radius: tuple[float, float] = (0.15, 0.32)  # [m]
    goal_max_bearing_deg: float = 45.0
    success_radius: float = 0.03  # block centre within this of the goal (xy) [m]
    # Reward = reach_weight * (1 - tanh(d_tip_pushpoint / 5 cm))
    #        + progress_weight * (d_block_goal_prev - d_block_goal)
    #        - push_linear * d_block_goal + push_weight * (1 - tanh(d_block_goal / 5 cm))
    #        + success_bonus * [success] - action_penalty * |a|^2
    # The push point is behind the block on the side away from the goal, at block-centre
    # height, push_point_offset from the block centre. Rewarding "near the block" instead
    # taught the first pilot to touch the block and shove it in random directions.
    # "v2" is the reward above. "v1" is the first design, kept as a baseline: its reach term
    # uses the distance to the block centre instead of the push point, it has no progress
    # term, and it is used with push_linear = 3 (see PushConfig.v1()).
    reward_version: str = "v2"
    reach_weight: float = 0.5
    push_point_offset: float = 0.035  # [m]
    progress_weight: float = 50.0  # per metre of block progress toward the goal, each step
    push_linear: float = 1.0
    push_weight: float = 1.0
    reward_scale: float = 0.05  # [m]
    success_bonus: float = 1.0
    action_penalty: float = 0.01

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def v1(cls) -> PushConfig:
        """The first push reward design (6% after 1.5M steps), for baseline comparisons."""
        return cls(reward_version="v1", push_linear=3.0)

    @classmethod
    def from_dict(cls, d: dict) -> PushConfig:
        d = dict(d)
        # Configs saved before reward versions existed: v1 had no progress term.
        d.setdefault("reward_version", "v2" if "progress_weight" in d else "v1")
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})
