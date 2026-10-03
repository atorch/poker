"""
Tests for poker.benchmark: duplicate dealing, paired decks, style statistics, the M1 check, reports.
"""
import json
import os
import random

import numpy as np
import pytest

from poker.benchmark import (
    M1_CRITERIA, agent_factory, check_m1, chips_per_deal_duplicate, format_report, run_benchmark,
)


def test_duplicate_dealing_scores_identical_agents_exactly_zero():
    for name in ["maxraise", "tag"]:
        make = agent_factory(name)
        mean, ci = chips_per_deal_duplicate(make, make, 50, random.Random(0))
        assert mean == 0 and ci == 0


def test_decks_depend_only_on_the_deck_rng():
    """The same deck rng deals the same decks whatever the agents do (paired comparisons)."""
    rngs = []
    for agent, opponent in [("random", "callstation"), ("maxraise", "tag"), ("tag-calldown", "random")]:
        rng = random.Random(1)
        np.random.seed(len(rngs))  # agents' own randomness differs between the runs
        chips_per_deal_duplicate(agent_factory(agent), agent_factory(opponent), 20, rng)
        rngs.append(rng.getstate())
    assert rngs[0] == rngs[1] == rngs[2]


def test_style_stats_of_card_blind_bots():
    report = run_benchmark("callstation", opponents=["maxraise"], n_decks=30, fingerprint=False)
    style = report["opponents"]["maxraise"]["style"]
    assert style["pfr"] == 0 and style["afq"] == 0 and style["fold_to_bet"] == 0
    assert style["wtsd"] == 1  # nobody ever folds

    report = run_benchmark("maxraise", opponents=["callstation"], n_decks=30, fingerprint=False)
    style = report["opponents"]["callstation"]["style"]
    assert style["vpip"] == 1 and style["pfr"] == 1 and style["fold_to_bet"] == 0
    assert style["afq"] == 1

    report = run_benchmark("tag", opponents=["maxraise"], n_decks=30, fingerprint=False)
    style = report["opponents"]["maxraise"]["style"]
    assert 0 < style["fold_to_bet"] < 1 and 0 < style["vpip"] < 1


def test_results_do_not_depend_on_panel_order():
    a = run_benchmark("tag", opponents=["random", "consistent"], n_decks=20, fingerprint=False)
    b = run_benchmark("tag", opponents=["consistent", "random"], n_decks=20, fingerprint=False)
    for name in ["random", "consistent"]:
        assert a["opponents"][name]["chips_per_deal"] == b["opponents"][name]["chips_per_deal"]


def test_benchmark_restores_global_random_state():
    random.seed(123)
    np.random.seed(123)
    expected = (random.random(), np.random.random())
    random.seed(123)
    np.random.seed(123)
    run_benchmark("random", opponents=["random"], n_decks=3, fingerprint=False)
    assert (random.random(), np.random.random()) == expected


def test_check_m1():
    passing = {name: {"chips_per_deal": 1.0, "chips_ci": 0.1} for name in M1_CRITERIA}
    assert check_m1(passing)["pass"]

    failing = dict(passing, maxraise={"chips_per_deal": 0.4, "chips_ci": 0.1})
    result = check_m1(failing)
    assert not result["pass"] and result["passed"] == len(M1_CRITERIA) - 1 and not result["checks"]["maxraise"]

    # A "CI > 0" criterion fails when the interval includes 0
    assert not check_m1(dict(passing, tag={"chips_per_deal": 0.1, "chips_ci": 0.2}))["checks"]["tag"]

    partial = {"maxraise": passing["maxraise"]}
    result = check_m1(partial)
    assert result["checked"] == 1 and not result["pass"]  # not every criterion was run


def test_report_is_json_serializable_and_compact():
    report = run_benchmark("tag", opponents=["callstation", "maxraise"], n_decks=10, n_games=2)
    json.dumps(report)
    lines = format_report(report)
    assert len(lines) <= 15
    m1_line = next(line for line in lines if line.startswith("M1: "))
    assert m1_line.startswith("M1: FAIL") and m1_line.endswith("of 2 criteria met (5 not run)")
    assert any("bust-out" in line for line in lines)


@pytest.mark.skipif(not os.path.exists("models/player_0_latest.h5"), reason="no saved model (models/ is gitignored)")
def test_model_spec_plays_from_every_seat():
    report = run_benchmark("models/player_0_latest.h5", opponents=["callstation", "models/player_0_latest.h5"],
                           n_decks=5, fingerprint=False)
    assert set(report["opponents"]) == {"callstation", "models/player_0_latest.h5"}
    assert report["agent"] == "player_0_latest.h5"
