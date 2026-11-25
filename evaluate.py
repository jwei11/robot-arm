"""Evaluate a trained policy: success rate, final distance, and an MP4 of a few rollouts.

Example (PowerShell):
  .\\.venv\\Scripts\\python.exe evaluate.py --model runs\\ppo_reach\\best_model.zip --episodes 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio
import numpy as np
from stable_baselines3 import PPO

from so101_rl.config import DomainRandomization, EnvConfig
from so101_rl.tasks import config_from_dict, env_class_for


def load_env_config(model_path: Path):
    for d in (model_path.parent, model_path.parent.parent):
        p = d / "env_config.json"
        if p.exists():
            return config_from_dict(json.loads(p.read_text()))
    return EnvConfig()


def evaluate(model, cfg, episodes: int, seed: int, video: Path | None, video_episodes: int) -> dict:
    env = env_class_for(cfg)(cfg, render_mode="rgb_array" if video else None)
    final_d, success, first_hit = [], [], []
    frames = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        record = video is not None and ep < video_episodes
        hit = None
        for t in range(cfg.episode_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            if hit is None and info["is_success"]:
                hit = t + 1
            if record:
                frames.append(env.render())
            if term or trunc:
                break
        final_d.append(info["distance"])
        success.append(info["is_success"])
        first_hit.append(hit)
    env.close()
    hits = [h for h in first_hit if h is not None]
    res = {
        "episodes": episodes,
        "success_rate": float(np.mean(success)),
        "mean_final_distance_cm": float(np.mean(final_d) * 100),
        "median_final_distance_cm": float(np.median(final_d) * 100),
        "p90_final_distance_cm": float(np.percentile(final_d, 90) * 100),
        "mean_steps_to_first_success": float(np.mean(hits)) if hits else None,
    }
    if video is not None and frames:
        video.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(video, frames, fps=env.metadata["render_fps"], macro_block_size=1)
        res["video"] = str(video)
    return res


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=100_000, help="eval seeds differ from training seeds")
    p.add_argument("--video", type=Path, default=None, help="default: <run dir>/rollouts.mp4")
    p.add_argument("--video-episodes", type=int, default=3)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--dr", action="store_true", help="evaluate with the sim2real randomization preset")
    args = p.parse_args(argv)

    cfg = load_env_config(args.model)
    if args.dr:
        if not isinstance(cfg, EnvConfig):
            raise SystemExit("--dr applies to the reach task only")
        cfg.dr = DomainRandomization.sim2real()
    model = PPO.load(str(args.model), device="cpu")
    video = None if args.no_video else (args.video or args.model.parent / "rollouts.mp4")
    res = evaluate(model, cfg, args.episodes, args.seed, video, args.video_episodes)
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main()
