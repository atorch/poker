"""
Train a poker.dqn agent against fixed opponents with Monte-Carlo targets
(TICKET_learner_redesign.md, M0 / phase A).

Each table plays single deals with random stacks, dealer and seat; the learner's decisions are
batched across tables and chosen epsilon-greedily over Q (in big blinds). With --targets mc
(the default), every non-fold decision of a finished deal becomes a replay sample whose target
is the deal's reward; with --targets td, replay holds transitions to the learner's next decision
and targets are Expected SARSA bootstrapped from a target network.

Writes a run directory (poker.runs): one console line every --log-every decisions, a benchmark
(poker.benchmark, fixed seed so evals are paired across checkpoints and runs) and a checkpoint
every --eval-every decisions, and final.pt plus a larger final benchmark at the end.

    uv run python -m poker.train_dqn --name mc_callstation --opponents callstation
    uv run python -m poker.train_dqn --name mc_pool --opponents pool --seeds 0,1,2      # every scripted agent
    uv run python -m poker.train_dqn --name ladder --curriculum "200000=callstation;200000=maxraise;600000=pool"
    uv run python -m poker.runs summarize runs/<run directory>
"""
import argparse
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from poker.actions import Head
from poker.benchmark import DEFAULT_PANEL, SeatedAgent, run_benchmark
from poker.dqn import (
    DealRecorder, DQNAgent, Learner, ReplayBuffer, TransitionBuffer, TransitionRecorder, epsilon_greedy,
    full_q_values, save_model,
)
from poker.env import VectorEnv
from poker.runs import RUNS_ROOT, RunLogger

DEFAULTS = {
    "opponents": "callstation",
    "curriculum": None,  # e.g. "200000=callstation;200000=maxraise;600000=pool" (overrides opponents and decisions)
    "decisions": 1_000_000,
    "tables": 256,
    "hidden": [256, 256],
    "learning_rate": 3e-4,
    "targets": "td",  # "td": Expected SARSA with a target network; "mc": regress toward the deal's reward
    "target_tau": 0.01,  # Polyak averaging rate of the target network (td only), per gradient step
    "batch_size": 256,
    "buffer_size": 100_000,
    "min_buffer": 5_000,
    "updates_per_decision": 0.0625,  # 16 replayed samples per decision at batch 256
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay_decisions": 300_000,
    "stack_min": 5,
    "stack_max": 35,
    "log_every": 50_000,
    "eval_every": 200_000,
    "eval_decks": 300,
    "final_eval_decks": 1000,
    "eval_opponents": list(DEFAULT_PANEL),
    "eval_seed": 0,
    "max_minutes": None,
    "threads": 2,
    "seed": 0,
}

Q_WARNING_BB = 50.0  # no deal can win or lose this much, so larger Q-values mean divergence


def parse_curriculum(text):
    """'200000=callstation;600000=pool' -> [(200000, 'callstation'), (600000, 'pool')]."""
    phases = []
    for item in text.split(";"):
        decisions, _, opponents = item.strip().partition("=")
        phases.append((int(decisions), opponents.strip()))
    return phases


def epsilon_at(config, decisions):
    fraction = min(1.0, decisions / max(1, config["epsilon_decay_decisions"]))
    return config["epsilon_start"] + fraction * (config["epsilon_end"] - config["epsilon_start"])


def evaluate(network, config, n_decks, step):
    make_agent = lambda seat: SeatedAgent(DQNAgent(network), seat)
    return run_benchmark(make_agent, opponents=config["eval_opponents"], n_decks=n_decks, n_games=0,
                         seed=config["eval_seed"], agent_name=f"step {step:,}")


def train(config, name="dqn", root=RUNS_ROOT, print_fn=None):
    """Train with `config` (DEFAULTS overridden by the given keys); returns the run directory."""
    config = {**DEFAULTS, **config}
    phases = parse_curriculum(config["curriculum"]) if config["curriculum"] else [(config["decisions"], config["opponents"])]
    config["decisions"] = sum(decisions for decisions, _ in phases)
    phase_ends = np.cumsum([decisions for decisions, _ in phases])
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["threads"])
    rng = np.random.default_rng(config["seed"])

    env = VectorEnv(n_tables=config["tables"], opponents=phases[0][1],
                    stack_range=(config["stack_min"], config["stack_max"]), seed=config["seed"])
    learner = Learner(hidden=config["hidden"], learning_rate=config["learning_rate"])
    use_td = config["targets"] == "td"
    assert config["targets"] in ("mc", "td"), config["targets"]
    buffer = TransitionBuffer(config["buffer_size"]) if use_td else ReplayBuffer(config["buffer_size"])
    recorder = TransitionRecorder(env.n_tables) if use_td else DealRecorder(env.n_tables)

    console_keys = ["loss", "reward_bb", "eps", "fold", "call", "raise", "q_mean", "dps"]
    with RunLogger(name, config, defaults=DEFAULTS, root=root, console_keys=console_keys, print_fn=print_fn) as run:
        start = time.time()
        decisions = steps = 0
        next_log, next_eval = config["log_every"], config["eval_every"]
        interval = {"losses": [], "rewards": [], "heads": [], "q": [], "start": time.time(), "decisions": 0,
                    "skipped": env.skipped_deals, "deals": env.deals_finished}
        stopped_early = False
        phase = 0

        while decisions < config["decisions"]:
            if decisions >= phase_ends[phase] and phase + 1 < len(phases):
                phase += 1
                env.set_opponents(phases[phase][1])
                run.print(f"phase {phase + 1}/{len(phases)} from step {decisions:,}: opponents {phases[phase][1]}")
            if config["max_minutes"] is not None and time.time() - start > 60 * config["max_minutes"]:
                stopped_early = True
                break

            batch = env.observe()
            if use_td:
                buffer.add(recorder.complete(batch))
            q = full_q_values(learner.network.learned_q(batch["features"]), batch["fold_values"], batch["masks"])
            epsilon = epsilon_at(config, decisions)
            heads = epsilon_greedy(q, batch["masks"], epsilon, rng)
            recorder.record(batch["features"], heads)
            rewards, done = env.step(heads)
            if use_td:
                buffer.add(recorder.finish(rewards, done))
            else:
                buffer.add(*recorder.finish(rewards, done))
            decisions += env.n_tables

            interval["decisions"] += env.n_tables
            interval["rewards"].extend(rewards[done].tolist())
            interval["heads"].extend(heads.tolist())
            interval["q"].append(float(q.max(axis=1).mean()))

            while len(buffer) >= config["min_buffer"] and steps < decisions * config["updates_per_decision"]:
                if use_td:
                    sample = buffer.sample(config["batch_size"], rng)
                    targets = learner.expected_sarsa_targets(sample, epsilon)
                    interval["losses"].append(learner.train_step(sample["features"], sample["heads"], targets))
                    learner.soft_update(config["target_tau"])
                else:
                    interval["losses"].append(learner.train_step(*buffer.sample(config["batch_size"], rng)))
                steps += 1

            if decisions >= next_log:
                next_log += config["log_every"]
                heads_seen = np.array(interval["heads"])
                deals = env.deals_finished - interval["deals"]
                skipped = env.skipped_deals - interval["skipped"]
                q_mean = float(np.mean(interval["q"]))
                run.log_metrics(decisions, **({"phase": phase + 1} if len(phases) > 1 else {}), **{
                    "loss": float(np.mean(interval["losses"])) if interval["losses"] else None,
                    "reward_bb": float(np.mean(interval["rewards"])) if interval["rewards"] else None,
                    "eps": epsilon,
                    "fold": float(np.mean(heads_seen == Head.FOLD)),
                    "call": float(np.mean(heads_seen == Head.CHECK_CALL)),
                    "raise": float(np.mean(heads_seen >= Head.RAISE_1)),
                    "q_mean": q_mean,
                    "deals": deals,
                    "skipped": skipped / max(1, deals + skipped),
                    "buffer": len(buffer),
                    "steps": steps,
                    "dps": interval["decisions"] / (time.time() - interval["start"]),
                })
                if abs(q_mean) > Q_WARNING_BB:
                    run.warn(f"mean greedy Q is {q_mean:.1f} BB (impossible in one deal)")
                interval = {"losses": [], "rewards": [], "heads": [], "q": [], "start": time.time(), "decisions": 0,
                            "skipped": env.skipped_deals, "deals": env.deals_finished}

            # Note: the last periodic eval is replaced by the (larger) final one
            if decisions >= next_eval and decisions < config["decisions"]:
                next_eval += config["eval_every"]
                run.log_eval(decisions, evaluate(learner.network, config, config["eval_decks"], decisions))
                save_model(learner.network, run.checkpoint_path(decisions, suffix=""), step=decisions)

        final_path = save_model(learner.network, run.dir / "final", step=decisions)
        run.log_eval(decisions, evaluate(learner.network, config, config["final_eval_decks"], decisions))
        run.finish(status="finished", model=str(final_path), stopped_early=stopped_early, decisions=decisions,
                   gradient_steps=steps)
    return run.dir


def launch_seeds(argv, seeds, root, name):
    """Run one training process per seed (same arguments otherwise), wait, and print a comparison."""
    from poker.runs import compare, load_run

    consoles = Path(root) / "consoles"
    consoles.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    processes = []
    for seed in seeds:
        console = consoles / f"{stamp}_{name}_seed{seed}.txt"
        with open(console, "w") as out:
            command = [sys.executable, "-m", "poker.train_dqn", *argv, "--seed", str(seed)]
            processes.append((subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT), console))
    print(f"Launched seeds {','.join(map(str, seeds))}; consoles in {consoles}/", flush=True)

    run_dirs = []
    for process, console in processes:
        process.wait()
        lines = console.read_text().splitlines()
        run_dir = next((line.split(": ", 1)[1] for line in lines if line.startswith("Run directory: ")), None)
        if process.returncode != 0 or run_dir is None:
            print(f"FAILED ({process.returncode}): see {console}")
        if run_dir is not None:
            run_dirs.append(run_dir)
    print("\n".join(compare([load_run(path) for path in run_dirs])))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="dqn", help="run name (the run directory is runs/<date>_<name>)")
    parser.add_argument("--root", default=RUNS_ROOT)
    parser.add_argument("--seeds", type=lambda text: [int(seed) for seed in text.split(",")],
                        help="run one process per seed, e.g. 0,1,2, then print poker.runs compare")
    for key, default in DEFAULTS.items():
        flag = "--" + key.replace("_", "-")
        if key == "hidden":
            parser.add_argument(flag, type=lambda text: [int(width) for width in text.split(",")], default=default,
                                help="layer widths, e.g. 256,256")
        elif key == "eval_opponents":
            parser.add_argument(flag, nargs="+", default=default)
        elif key == "max_minutes":
            parser.add_argument(flag, type=float, default=default)
        elif key == "curriculum":
            parser.add_argument(flag, default=default, help='e.g. "200000=callstation;600000=pool"')
        else:
            parser.add_argument(flag, type=type(default), default=default)
    args = vars(parser.parse_args())
    name, root, seeds = args.pop("name"), args.pop("root"), args.pop("seeds")
    if seeds:
        argv = list(sys.argv[1:])
        index = next(i for i, token in enumerate(argv) if token.startswith("--seeds"))
        del argv[index:index + (1 if "=" in argv[index] else 2)]
        launch_seeds(argv, seeds, root, name)
    else:
        train(args, name=name, root=root)


if __name__ == "__main__":
    main()
