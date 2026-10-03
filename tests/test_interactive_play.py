"""
Smoke test for interactive_play.py: a scripted "human" plays a short game against each kind of agent.
"""
import builtins
import itertools

import pytest
import torch

import interactive_play
from poker.dqn import QNetwork, save_model


@pytest.mark.parametrize("full_info", [False, True])
def test_short_game_against_a_model_and_a_scripted_bot(tmp_path, monkeypatch, capsys, full_info):
    torch.manual_seed(0)
    model = save_model(QNetwork(hidden=(16,)), tmp_path / "model")
    # Note: "-1" folds when facing a bet; otherwise it is rejected and "0" checks
    answers = itertools.cycle(["-1", "0"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(answers))

    for spec in [str(model), "tag"]:
        winner = interactive_play.play_interactive_game(model_path=spec, full_info=full_info, max_deals=3)
        assert winner in (0, 1, 2)

    output = capsys.readouterr().out
    assert "GAME OVER" in output and "Traceback" not in output
    assert ("Policy]" in output) == full_info
