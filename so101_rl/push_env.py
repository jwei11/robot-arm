"""Gymnasium environment: SO-101 pushes a 4 cm cube across the table onto a goal disc.

Same arm, action interface and control rate as the reach task (delta joint targets for
the 5 arm joints, gripper held closed, 20 Hz). What is new is contact: the arm has to get
behind the block and push it, so the policy must learn where to stand relative to the
block and goal, not just where to put its tip. Sim only: run_real.py does not run it.
"""

from __future__ import annotations

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from so101_rl.config import PushConfig
from so101_rl.env import DELTA_SCALE, VEL_SCALE, apply_action
from so101_rl.kinematics import ARM_JOINTS, ASSET_DIR, EE_SITE, GRIPPER_JOINT, Kinematics, load_model

PUSH_SCENE_XML = ASSET_DIR / "push_scene.xml"
N_ARM = len(ARM_JOINTS)
BLOCK_HALF = 0.02  # [m], matches push_scene.xml

PUSH_OBS_LAYOUT: dict[str, slice] = {
    "qpos": slice(0, 5),
    "qvel": slice(5, 10),  # finite-difference joint velocity * VEL_SCALE
    "ee_pos": slice(10, 13),
    "block_pos": slice(13, 16),
    "block_vel": slice(16, 19),  # finite-difference block velocity * VEL_SCALE
    "goal_xy": slice(19, 21),
    "ee_to_block": slice(21, 24),  # (block - ee) * DELTA_SCALE
    "block_to_goal": slice(24, 26),  # (goal - block)_xy * DELTA_SCALE
    "prev_action": slice(26, 31),
}
PUSH_OBS_DIM = 31


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def _polar(rng: np.random.Generator, radius: tuple[float, float], max_bearing_deg: float) -> np.ndarray:
    r = rng.uniform(*radius)
    b = np.radians(rng.uniform(-max_bearing_deg, max_bearing_deg))
    return np.array([r * np.cos(b), r * np.sin(b)])


class SO101PushEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, config: PushConfig | None = None, render_mode: str | None = None):
        self.cfg = config if config is not None else PushConfig()
        self.render_mode = render_mode
        self.model = load_model(PUSH_SCENE_XML)
        self.data = mujoco.MjData(self.model)
        self.kin = Kinematics(self.model, gripper_q=self.cfg.gripper_q)

        m = self.model
        self.n_substeps = int(round(1.0 / (self.cfg.control_hz * m.opt.timestep)))
        self.dt = self.n_substeps * m.opt.timestep
        self.metadata = {**self.metadata, "render_fps": int(round(1.0 / self.dt))}

        jids = [m.joint(n).id for n in ARM_JOINTS]
        self.qpos_adr = np.array([m.jnt_qposadr[j] for j in jids])
        self.act_ids = np.array([m.actuator(n).id for n in ARM_JOINTS])
        self.gripper_qadr = m.jnt_qposadr[m.joint(GRIPPER_JOINT).id]
        self.gripper_act = m.actuator(GRIPPER_JOINT).id
        self.ee_site = m.site(EE_SITE).id
        self.block_qadr = m.jnt_qposadr[m.joint("block_joint").id]
        self.block_body = m.body("block").id
        self.goal_mocap = m.body_mocapid[m.body("goal").id]
        # Goal distance range from the block [m]; a training curriculum may narrow it.
        self.goal_offset = tuple(self.cfg.goal_offset)

        lim = m.jnt_range[jids]
        self.q_low = lim[:, 0] + self.cfg.limit_margin
        self.q_high = lim[:, 1] - self.cfg.limit_margin

        self.action_space = spaces.Box(-1.0, 1.0, shape=(N_ARM,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(PUSH_OBS_DIM,), dtype=np.float32)

        self._renderer: mujoco.Renderer | None = None
        self.goal_xy = np.zeros(2)
        self.q_target = np.zeros(N_ARM)
        self.prev_action = np.zeros(N_ARM)
        self.q_prev = np.zeros(N_ARM)
        self.block_prev = np.zeros(3)
        self.step_count = 0

    # ------------------------------------------------------------------ state
    def ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.ee_site].copy()

    def block_pos(self) -> np.ndarray:
        return self.data.xpos[self.block_body].copy()

    def _arm_touches_block(self) -> bool:
        for c in self.data.contact[: self.data.ncon]:
            bodies = {self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]}
            if self.block_body in bodies and 0 not in bodies:
                return True
        return False

    def _get_obs(self) -> np.ndarray:
        q = self.data.qpos[self.qpos_adr].copy()
        ee, block = self.ee_pos(), self.block_pos()
        obs = np.concatenate(
            [
                q,
                (q - self.q_prev) / self.dt * VEL_SCALE,
                ee,
                block,
                (block - self.block_prev) / self.dt * VEL_SCALE,
                self.goal_xy,
                (block - ee) * DELTA_SCALE,
                (self.goal_xy - block[:2]) * DELTA_SCALE,
                self.prev_action,
            ]
        ).astype(np.float32)
        self.q_prev, self.block_prev = q, block
        return obs

    # ---------------------------------------------------------------- gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        rng, c = self.np_random, self.cfg
        for _ in range(100):
            mujoco.mj_resetData(self.model, self.data)
            home = np.asarray(c.home_qpos, dtype=np.float64)
            q0 = np.clip(home + rng.uniform(-c.home_noise, c.home_noise, N_ARM), self.q_low, self.q_high)
            self.data.qpos[self.qpos_adr] = q0
            self.data.qpos[self.gripper_qadr] = c.gripper_q
            self.data.ctrl[self.act_ids] = q0
            self.data.ctrl[self.gripper_act] = c.gripper_q

            block_xy = np.asarray(options.get("block_xy", _polar(rng, c.block_radius, c.block_max_bearing_deg)))
            yaw = np.radians(rng.uniform(-c.block_yaw_range_deg, c.block_yaw_range_deg))
            self.data.qpos[self.block_qadr : self.block_qadr + 7] = [*block_xy, BLOCK_HALF, *_yaw_quat(yaw)]
            if "goal_xy" in options:
                self.goal_xy = np.asarray(options["goal_xy"], dtype=np.float64)
            else:
                self.goal_xy = self._sample_goal(rng, block_xy)
            self.data.mocap_pos[self.goal_mocap] = [*self.goal_xy, 0.001]
            mujoco.mj_forward(self.model, self.data)
            if not self._arm_touches_block():
                break
        self.q_target = q0.copy()
        self.step_count = 0
        self.prev_action = np.zeros(N_ARM)
        self.q_prev = q0.copy()
        self.block_prev = self.block_pos()
        info = self._info()
        self._d_bg_prev = info["distance"]
        return self._get_obs(), info

    def _sample_goal(self, rng: np.random.Generator, block_xy: np.ndarray) -> np.ndarray:
        c = self.cfg
        for _ in range(1000):
            ang = rng.uniform(-np.pi, np.pi)
            g = block_xy + rng.uniform(*self.goal_offset) * np.array([np.cos(ang), np.sin(ang)])
            r, b = np.hypot(*g), np.degrees(np.arctan2(g[1], g[0]))
            if c.goal_radius[0] <= r <= c.goal_radius[1] and abs(b) <= c.goal_max_bearing_deg:
                return g
        raise RuntimeError("Could not sample a goal; check PushConfig ranges.")

    def push_point(self) -> np.ndarray:
        block = self.block_pos()
        to_goal = self.goal_xy - block[:2]
        n = np.linalg.norm(to_goal)
        u = to_goal / n if n > 1e-6 else np.zeros(2)
        return np.array([*(block[:2] - u * self.cfg.push_point_offset), block[2]])

    def set_goal_offset(self, low: float, high: float) -> None:
        """Change how far from the block new goals are placed (used by the curriculum)."""
        self.goal_offset = (float(low), float(high))

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.q_target = apply_action(self.q_target, a, self.cfg.max_delta, self.q_low, self.q_high)
        self.data.ctrl[self.act_ids] = self.q_target
        self.data.ctrl[self.gripper_act] = self.cfg.gripper_q
        mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
        mujoco.mj_kinematics(self.model, self.data)
        self.step_count += 1

        d_prev = self._d_bg_prev
        info = self._info()
        d_reach = info["ee_to_push_point"] if self.cfg.reward_version == "v2" else info["ee_to_block"]
        reward = self.compute_reward(d_reach, info["distance"], d_prev, a)
        self._d_bg_prev = info["distance"]
        self.prev_action = a
        truncated = self.step_count >= self.cfg.episode_steps
        return self._get_obs(), reward, False, truncated, info

    def compute_reward(self, d_ep: float, d_bg: float, d_bg_prev: float, action: np.ndarray) -> float:
        c = self.cfg
        reach = c.reach_weight * (1.0 - np.tanh(d_ep / c.reward_scale))
        progress = c.progress_weight * (d_bg_prev - d_bg) if c.reward_version == "v2" else 0.0
        push = progress - c.push_linear * d_bg + c.push_weight * (1.0 - np.tanh(d_bg / c.reward_scale))
        bonus = c.success_bonus if d_bg < c.success_radius else 0.0
        return float(reach + push + bonus - c.action_penalty * float(np.sum(np.square(action))))

    def _info(self) -> dict:
        block = self.block_pos()
        d_bg = float(np.linalg.norm(block[:2] - self.goal_xy))
        d_eb = float(np.linalg.norm(block - self.ee_pos()))
        d_ep = float(np.linalg.norm(self.push_point() - self.ee_pos()))
        return {
            "distance": d_bg,
            "is_success": d_bg < self.cfg.success_radius,
            "ee_to_block": d_eb,
            "ee_to_push_point": d_ep,
        }

    # ----------------------------------------------------------------- render
    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 480, 640)
            self._camera = mujoco.MjvCamera()
            self._camera.lookat[:] = [0.2, 0.0, 0.05]
            self._camera.distance, self._camera.azimuth, self._camera.elevation = 0.8, 150, -35
        self._renderer.update_scene(self.data, camera=self._camera)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
