"""End-to-end smoke tests: train.py runs for each task and writes a loadable model."""

import json

import pytest

import train


@pytest.mark.parametrize(
    "extra",
    [["--task", "reach"], ["--task", "push"], ["--task", "push", "--curriculum"], ["--task", "push", "--reward", "v1"]],
)
def test_train_runs_and_saves(tmp_path, extra):
    argv = [
        *extra,
        "--timesteps", "128",
        "--n-envs", "1",
        "--n-steps", "64",
        "--batch-size", "64",
        "--n-epochs", "1",
        "--eval-freq", "1000000",
        "--checkpoint-freq", "1000000",
        "--logdir", str(tmp_path),
        "--run-name", "smoke",
    ]  # fmt: skip
    run_dir = train.main(argv)
    assert (run_dir / "final_model.zip").exists()
    saved = json.loads((run_dir / "env_config.json").read_text())
    assert saved["task"] == extra[1]
    if "v1" in extra:
        assert saved["reward_version"] == "v1" and saved["push_linear"] == 3.0
    if "--curriculum" in extra:
        assert json.loads((run_dir / "curriculum.json").read_text())[0]["stage"] == 0


def test_curriculum_rejected_for_reach(tmp_path):
    with pytest.raises(SystemExit):
        train.main(["--task", "reach", "--curriculum", "--logdir", str(tmp_path)])
