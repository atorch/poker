"""
End-to-end tests for poker.train_dqn: a tiny run writes a complete run directory and a loadable model.
"""
from poker.dqn import load_agent
from poker.runs import load_run, summarize
from poker.state import State
from poker.train_dqn import train

TINY = {
    "decisions": 2000, "tables": 16, "hidden": [32], "batch_size": 32, "min_buffer": 200,
    "epsilon_decay_decisions": 1000, "log_every": 500, "eval_every": 1000, "eval_decks": 2,
    "final_eval_decks": 3, "eval_opponents": ["callstation", "tag"],
}


def test_tiny_training_run(tmp_path):
    printed = []
    run_dir = train(TINY, name="smoke", root=tmp_path, print_fn=printed.append)

    run = load_run(run_dir)
    assert run["summary"]["status"] == "finished" and run["summary"]["decisions"] >= 2000
    assert len(run["metrics"]) == 4 and {"loss", "reward_bb", "fold", "call", "raise"} <= set(run["metrics"][-1])
    assert [record["n_decks"] for record in run["evals"]] == [2, 3]  # one periodic eval, then the final one
    assert (run_dir / "final.pt").exists() and list((run_dir / "checkpoints").glob("*.pt"))
    assert len(printed) <= 12

    agent = load_agent(run_dir / "final.pt")
    state = State(n_players=3, initial_wealth=20, initial_dealer=0)
    assert agent.get_action(state) in state.legal_actions()
    assert len(summarize(run)) <= 30


def test_learns_to_use_its_cards_against_a_calling_station(tmp_path):
    """
    Learning sanity check (TICKET_learner_redesign.md §8): a short run against CallStation learns to
    fold weak hands more than strong ones, and that calling with AA beats folding it. Four seeds at
    this budget all gave fold spreads of +0.42 to +0.73 and Q(call | AA) of +3.6 to +5.9 BB.
    """
    import random

    from poker.actions import Head
    from poker.benchmark import SeatedAgent
    from poker.cards import parse_cards, stacked_deck
    from poker.diagnostics import fingerprint
    from poker.dqn import DQNAgent, load_network

    config = {"opponents": "callstation", "targets": "mc", "decisions": 30000, "tables": 64, "hidden": [64], "learning_rate": 1e-3,
              "min_buffer": 2000, "epsilon_decay_decisions": 15000, "log_every": 10**9, "eval_every": 10**9,
              "final_eval_decks": 2, "eval_opponents": ["callstation"], "seed": 0}
    network = load_network(train(config, name="sanity", root=tmp_path, print_fn=lambda line: None) / "final.pt")

    preflop = fingerprint(lambda seat: SeatedAgent(DQNAgent(network), seat))["preflop"]
    assert preflop["fold_spread"] > 0.2 and preflop["raise_spread"] > 0
    assert preflop["by_class"]["72o"][0] > preflop["by_class"]["AA"][0]  # P(fold)

    deck = stacked_deck([parse_cards("Ah Ad"), None, None], rng=random.Random(0))
    q = DQNAgent(network).head_values(State(n_players=3, initial_wealth=20, initial_dealer=0, deck=deck))
    assert q[Head.CHECK_CALL] > q[Head.FOLD] + 1


def test_curriculum_switches_opponents(tmp_path):
    printed = []
    config = dict(TINY, curriculum="1000=callstation;1000=maxraise", eval_every=10**9)
    run = load_run(train(config, name="ladder", root=tmp_path, print_fn=printed.append))
    assert [record["phase"] for record in run["metrics"]] == [1, 1, 2, 2]
    assert any(line.startswith("phase 2/2 from step") for line in printed)
    assert run["summary"]["decisions"] >= 2000


def test_tiny_td_training_run(tmp_path):
    run = load_run(train(dict(TINY, targets="td"), name="td_smoke", root=tmp_path, print_fn=lambda line: None))
    assert run["summary"]["status"] == "finished" and all(r["loss"] is not None for r in run["metrics"][1:])
