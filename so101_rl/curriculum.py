"""Goal-distance curriculum for the push task.

Training starts with goals close to the block, where success happens often enough to learn
from, and moves them further away each time the recent training success rate passes a
threshold, until the full task range is reached. Evaluation always uses the full range.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


def stage_range(stage: int, n_stages: int, start: tuple[float, float], end: tuple[float, float]) -> tuple[float, float]:
    """Goal-offset range for a stage: linear from `start` (stage 0) to `end` (last stage)."""
    f = 1.0 if n_stages <= 1 else min(stage, n_stages - 1) / (n_stages - 1)
    return (start[0] + f * (end[0] - start[0]), start[1] + f * (end[1] - start[1]))


class GoalDistanceCurriculum(BaseCallback):
    def __init__(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        n_stages: int = 5,
        threshold: float = 0.5,
        window: int = 200,
        log_path: Path | None = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.start, self.end, self.n_stages = start, end, n_stages
        self.threshold, self.window = threshold, window
        self.log_path = log_path
        self.stage = 0
        self.outcomes: deque[bool] = deque(maxlen=window)
        self.history: list[dict] = []

    def _apply(self) -> None:
        lo, hi = stage_range(self.stage, self.n_stages, self.start, self.end)
        self.training_env.env_method("set_goal_offset", lo, hi)
        self.history.append({"stage": self.stage, "timestep": int(self.num_timesteps), "goal_offset": [lo, hi]})
        if self.log_path is not None:
            self.log_path.write_text(json.dumps(self.history, indent=2))
        if self.verbose:
            print(
                f"[curriculum] stage {self.stage}/{self.n_stages - 1} at {self.num_timesteps} steps: "
                f"goals {lo * 100:.1f}-{hi * 100:.1f} cm from the block"
            )

    def _on_training_start(self) -> None:
        self._apply()

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"], strict=True):
            if done:
                self.outcomes.append(bool(info.get("is_success", False)))
        rate = float(np.mean(self.outcomes)) if self.outcomes else 0.0
        self.logger.record("curriculum/stage", self.stage)
        self.logger.record("curriculum/train_success", rate)
        if len(self.outcomes) == self.window and rate >= self.threshold and self.stage < self.n_stages - 1:
            self.stage += 1
            self.outcomes.clear()
            self._apply()
        return True
