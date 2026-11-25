"""Model loading, joint metadata, and forward kinematics for the SO-101.

Everything here is taken from the official MJCF; nothing about the geometry is
hand-written. Used by the env, the target sampler, and run_real.py (which needs
the same FK to build observations from real joint readings).
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

ASSET_DIR = Path(__file__).resolve().parent.parent / "assets" / "so101"
SCENE_XML = ASSET_DIR / "reach_scene.xml"

# The 5 joints the policy controls, in action/observation order.
ARM_JOINTS: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
GRIPPER_JOINT = "gripper"
EE_SITE = "gripperframe"


def load_model(path: str | Path = SCENE_XML) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(path))


class Kinematics:
    """Pure kinematics (no dynamics) on a private MjData."""

    def __init__(self, model: mujoco.MjModel | None = None, gripper_q: float = 0.0):
        self.model = model if model is not None else load_model()
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.joint_ids = np.array([m.joint(n).id for n in ARM_JOINTS])
        self.qpos_adr = np.array([m.jnt_qposadr[j] for j in self.joint_ids])
        self.gripper_adr = m.jnt_qposadr[m.joint(GRIPPER_JOINT).id]
        self.ee_site = m.site(EE_SITE).id
        self.limits = m.jnt_range[self.joint_ids].copy()  # (5, 2) rad
        self.gripper_q = gripper_q

    def _set(self, q: np.ndarray) -> None:
        self.data.qpos[:] = 0.0
        self.data.qpos[self.qpos_adr] = q
        self.data.qpos[self.gripper_adr] = self.gripper_q

    def fk(self, q: np.ndarray) -> np.ndarray:
        """End-effector (gripperframe site) world position for 5 arm joint angles [rad]."""
        self._set(np.asarray(q, dtype=np.float64))
        mujoco.mj_kinematics(self.model, self.data)
        return self.data.site_xpos[self.ee_site].copy()

    def solve_ik(
        self,
        target: np.ndarray,
        low: np.ndarray,
        high: np.ndarray,
        rng: np.random.Generator | None = None,
        restarts: int = 20,
        iters: int = 200,
        tol: float = 1e-3,
    ) -> np.ndarray | None:
        """Position-only damped-least-squares IK within [low, high]. Returns a
        collision-free q with |fk(q) - target| < tol, or None. Used to verify
        that fixed targets (run_real.py) are reachable."""
        rng = rng if rng is not None else np.random.default_rng(0)
        target = np.asarray(target, float)
        jac = np.zeros((3, self.model.nv))
        dofs = np.array([self.model.jnt_dofadr[j] for j in self.joint_ids])
        for _ in range(restarts):
            q = rng.uniform(low, high)
            for _ in range(iters):
                err = target - self.fk(q)
                if np.linalg.norm(err) < tol:
                    break
                mujoco.mj_comPos(self.model, self.data)  # jacobians need cdof / subtree_com
                mujoco.mj_jacSite(self.model, self.data, jac, None, self.ee_site)
                j = jac[:, dofs]
                dq = j.T @ np.linalg.solve(j @ j.T + 1e-4 * np.eye(3), err)
                q = np.clip(q + dq, low, high)
            if np.linalg.norm(target - self.fk(q)) < tol and not self.in_collision(q):
                return q
        return None

    def in_collision(self, q: np.ndarray) -> bool:
        """True if the arm touches the floor or itself at configuration q."""
        self._set(np.asarray(q, dtype=np.float64))
        mujoco.mj_forward(self.model, self.data)
        return self.data.ncon > 0
