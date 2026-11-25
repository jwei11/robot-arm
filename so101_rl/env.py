"""Gymnasium environment: SO-101 gripper tip reaches a random 3D target in MuJoCo."""

from __future__ import annotations

from collections import deque

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from so101_rl.config import EnvConfig
from so101_rl.kinematics import ARM_JOINTS, EE_SITE, GRIPPER_JOINT, Kinematics, load_model
from so101_rl.targets import WorkspaceSampler, sample_reachable_target

N_ARM = len(ARM_JOINTS)
VEL_SCALE = 0.5  # observed joint velocity [rad/s] is multiplied by this
DELTA_SCALE = 5.0  # (target - ee) [m] is multiplied by this

# Observation layout (float32, 24 values). Shared with run_real.py via build_observation().
OBS_LAYOUT: dict[str, slice] = {
    "qpos": slice(0, 5),  # observed arm joint angles [rad]
    "qvel": slice(5, 10),  # finite-difference joint velocity [rad/s] * VEL_SCALE
    "ee_pos": slice(10, 13),  # FK of observed joints [m]
    "target_pos": slice(13, 16),  # [m]
    "ee_to_target": slice(16, 19),  # (target - ee) * DELTA_SCALE
    "prev_action": slice(19, 24),  # last policy action in [-1, 1]
}
OBS_DIM = 24


def build_observation(
    q_obs: np.ndarray,
    q_obs_prev: np.ndarray,
    dt: float,
    ee_pos: np.ndarray,
    target_pos: np.ndarray,
    prev_action: np.ndarray,
) -> np.ndarray:
    """Single source of truth for the observation vector (sim and real)."""
    qvel = (q_obs - q_obs_prev) / dt
    return np.concatenate(
        [
            q_obs,
            qvel * VEL_SCALE,
            ee_pos,
            target_pos,
            (target_pos - ee_pos) * DELTA_SCALE,
            prev_action,
        ]
    ).astype(np.float32)


def apply_action(
    q_target: np.ndarray, action: np.ndarray, max_delta: float, low: np.ndarray, high: np.ndarray
) -> np.ndarray:
    """Delta joint-position target: clip action to [-1, 1], scale, add, clamp to limits.
    Shared with run_real.py."""
    a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    return np.clip(q_target + a * max_delta, low, high)


class SO101ReachEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, config: EnvConfig | None = None, render_mode: str | None = None):
        self.cfg = config if config is not None else EnvConfig()
        self.render_mode = render_mode
        self.model = load_model()
        self.data = mujoco.MjData(self.model)
        self.kin = Kinematics(self.model, gripper_q=self.cfg.gripper_q)

        m = self.model
        self.n_substeps = int(round(1.0 / (self.cfg.control_hz * m.opt.timestep)))
        self.dt = self.n_substeps * m.opt.timestep
        self.metadata = {**self.metadata, "render_fps": int(round(1.0 / self.dt))}

        jids = [m.joint(n).id for n in ARM_JOINTS]
        self.qpos_adr = np.array([m.jnt_qposadr[j] for j in jids])
        self.qvel_adr = np.array([m.jnt_dofadr[j] for j in jids])
        self.dof_ids = self.qvel_adr.copy()
        self.act_ids = np.array([m.actuator(n).id for n in ARM_JOINTS])
        self.gripper_qadr = m.jnt_qposadr[m.joint(GRIPPER_JOINT).id]
        self.gripper_act = m.actuator(GRIPPER_JOINT).id
        self.ee_site = m.site(EE_SITE).id
        self.target_mocap = m.body_mocapid[m.body("target").id]

        lim = m.jnt_range[jids]
        self.q_low = lim[:, 0] + self.cfg.limit_margin
        self.q_high = lim[:, 1] - self.cfg.limit_margin
        self._workspace_sampler = None
        if self.cfg.target_sampling == "workspace":
            self._workspace_sampler = WorkspaceSampler(
                self.kin,
                self.q_low,
                self.q_high,
                self.cfg.target_min_z,
                self.cfg.target_min_radius,
                max_bearing_deg=self.cfg.target_max_bearing_deg,
            )
        elif self.cfg.target_sampling != "joint":
            raise ValueError(f"unknown target_sampling {self.cfg.target_sampling!r}")

        # Nominal physics params, restored before each domain-randomization sample.
        self._nominal = {
            "damping": m.dof_damping.copy(),
            "armature": m.dof_armature.copy(),
            "frictionloss": m.dof_frictionloss.copy(),
            "gainprm": m.actuator_gainprm.copy(),
            "biasprm": m.actuator_biasprm.copy(),
        }

        self.action_space = spaces.Box(-1.0, 1.0, shape=(N_ARM,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)

        self._renderer: mujoco.Renderer | None = None
        self.target_pos = np.zeros(3)
        self.q_target = np.zeros(N_ARM)
        self.prev_action = np.zeros(N_ARM)
        self.q_obs_prev = np.zeros(N_ARM)
        self.step_count = 0
        self.latency = 0
        self._action_queue: deque[np.ndarray] = deque()

    # ------------------------------------------------------------------ physics
    def _randomize_physics(self) -> None:
        m, nom, dr, rng = self.model, self._nominal, self.cfg.dr, self.np_random
        m.dof_damping[:] = nom["damping"]
        m.dof_armature[:] = nom["armature"]
        m.dof_frictionloss[:] = nom["frictionloss"]
        m.actuator_gainprm[:] = nom["gainprm"]
        m.actuator_biasprm[:] = nom["biasprm"]
        self.latency = 0
        if not dr.enabled:
            return
        n = m.nv
        m.dof_damping[:] = nom["damping"] * rng.uniform(*dr.damping_scale, size=n)
        m.dof_armature[:] = nom["armature"] * rng.uniform(*dr.armature_scale, size=n)
        m.dof_frictionloss[:] = nom["frictionloss"] * rng.uniform(*dr.frictionloss_scale, size=n)
        g = rng.uniform(*dr.actuator_gain_scale, size=m.nu)
        # Position actuator: force = kp*ctrl - kp*q - kv*qdot  (gainprm[0]=kp, biasprm[1]=-kp, biasprm[2]=-kv)
        m.actuator_gainprm[:, 0] = nom["gainprm"][:, 0] * g
        m.actuator_biasprm[:, 1] = nom["biasprm"][:, 1] * g
        m.actuator_biasprm[:, 2] = nom["biasprm"][:, 2] * g
        lo, hi = dr.action_latency_steps
        self.latency = int(rng.integers(lo, hi + 1))

    def _observe_q(self) -> np.ndarray:
        q = self.data.qpos[self.qpos_adr].copy()
        std = self.cfg.dr.obs_noise_std if self.cfg.dr.enabled else 0.0
        if std > 0:
            q = q + self.np_random.normal(0.0, std, size=q.shape)
        return q

    def ee_pos(self) -> np.ndarray:
        """True end-effector position (used for reward and success)."""
        return self.data.site_xpos[self.ee_site].copy()

    def _get_obs(self, q_obs: np.ndarray) -> np.ndarray:
        obs = build_observation(q_obs, self.q_obs_prev, self.dt, self.kin.fk(q_obs), self.target_pos, self.prev_action)
        self.q_obs_prev = q_obs
        return obs

    # ---------------------------------------------------------------- gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        rng = self.np_random
        self._randomize_physics()

        mujoco.mj_resetData(self.model, self.data)
        home = np.asarray(self.cfg.home_qpos, dtype=np.float64)
        q0 = np.clip(home + rng.uniform(-self.cfg.home_noise, self.cfg.home_noise, N_ARM), self.q_low, self.q_high)
        self.data.qpos[self.qpos_adr] = q0
        self.data.qpos[self.gripper_qadr] = self.cfg.gripper_q
        self.q_target = q0.copy()
        self.data.ctrl[self.act_ids] = q0
        self.data.ctrl[self.gripper_act] = self.cfg.gripper_q

        if "target_pos" in options:
            self.target_pos = np.asarray(options["target_pos"], dtype=np.float64)
        elif self._workspace_sampler is not None:
            self.target_pos, _ = self._workspace_sampler(rng)
        else:
            self.target_pos, _ = sample_reachable_target(
                self.kin,
                rng,
                self.q_low,
                self.q_high,
                self.cfg.target_min_z,
                self.cfg.target_min_radius,
                max_bearing_deg=self.cfg.target_max_bearing_deg,
            )
        self.data.mocap_pos[self.target_mocap] = self.target_pos
        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.prev_action = np.zeros(N_ARM)
        self._action_queue = deque([np.zeros(N_ARM)] * self.latency)
        q_obs = self._observe_q()
        self.q_obs_prev = q_obs  # zero velocity on the first observation
        obs = self._get_obs(q_obs)
        return obs, self._info()

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self._action_queue.append(a)
        applied = self._action_queue.popleft()  # delayed by `latency` steps

        self.q_target = apply_action(self.q_target, applied, self.cfg.max_delta, self.q_low, self.q_high)
        self.data.ctrl[self.act_ids] = self.q_target
        self.data.ctrl[self.gripper_act] = self.cfg.gripper_q
        mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
        mujoco.mj_kinematics(self.model, self.data)  # site_xpos is otherwise one substep stale
        self.step_count += 1

        dist = float(np.linalg.norm(self.ee_pos() - self.target_pos))
        reward = self.compute_reward(dist, a)
        self.prev_action = a
        obs = self._get_obs(self._observe_q())
        truncated = self.step_count >= self.cfg.episode_steps
        return obs, reward, False, truncated, self._info(dist)

    def compute_reward(self, dist: float, action: np.ndarray) -> float:
        c = self.cfg
        bonus = c.success_bonus if dist < c.success_radius else 0.0
        precision = c.precision_weight * (1.0 - np.tanh(dist / c.precision_scale))
        return float(-dist + bonus + precision - c.action_penalty * float(np.sum(np.square(action))))

    def _info(self, dist: float | None = None) -> dict:
        if dist is None:
            dist = float(np.linalg.norm(self.ee_pos() - self.target_pos))
        return {"distance": dist, "is_success": dist < self.cfg.success_radius, "latency": self.latency}

    # ----------------------------------------------------------------- render
    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 480, 640)
            self._camera = mujoco.MjvCamera()
            self._camera.lookat[:] = [0.12, 0.0, 0.17]
            self._camera.distance, self._camera.azimuth, self._camera.elevation = 1.1, 135, -22
        self._renderer.update_scene(self.data, camera=self._camera)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
