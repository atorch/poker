"""
End-to-end smoke test of the training pipeline with real TensorFlow.

Other tests mock poker.q_function (so they run without TensorFlow), which means they can't
catch breakage in the Keras calls themselves. This test runs a tiny training job in a
subprocess, so the mocks installed by other test modules can't leak into it.
"""
import os
import subprocess
import sys

TRAINING_SCRIPT = """
import sys
from poker.play import run_sarsa

model_path = sys.argv[1]
run_sarsa(
    n_players=3,
    n_episodes=4,
    model_path=model_path,
    save_interval=2,
    max_deals=30,
    curriculum_random_episodes=2,
    curriculum_mixed_episodes=1,
    hidden_layers=(8, 8),
    batch_size=4,
    skip_fairness_test=True,
    skip_equilibrium_test=True,
    random_seed=0,
    n_eval_episodes=3,
)
"""


def test_tiny_training_run_completes_and_saves_models(tmp_path):
    model_path = str(tmp_path / "model.weights.h5")
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    result = subprocess.run(
        [sys.executable, "-c", TRAINING_SCRIPT, model_path],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    saved = sorted(os.listdir(tmp_path))
    assert "model.weights.h5" in saved
    assert "model_ep2.weights.h5" in saved
    assert any(name.endswith("_ep4_final.weights.h5") for name in saved)
