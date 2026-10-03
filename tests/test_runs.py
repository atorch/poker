"""
Tests for poker.runs: the run directory written by RunLogger and the summarize/compare/list/plot tools.
"""
import json
import math
import subprocess

import pytest

from poker.benchmark import run_benchmark
from poker.runs import RunLogger, compare, list_runs, load_run, plot, run_status, summarize


@pytest.fixture(scope="module")
def report():
    return run_benchmark("tag", opponents=["maxraise", "callstation"], n_decks=5, fingerprint=False)


def write_run(root, report, seed=0, name="mc_test", printed=None):
    config = {"learning_rate": 0.001, "opponents": ["callstation"], "seed": seed}
    defaults = {"learning_rate": 0.0003, "opponents": ["callstation"], "seed": 0}
    print_fn = printed.append if printed is not None else (lambda line: None)
    with RunLogger(name, config, defaults=defaults, root=root, print_fn=print_fn) as run:
        for step in range(0, 1000, 100):
            run.log_metrics(step, loss=1.0 / (step + 1), q_abs=0.5, entropy=1.2 - step / 1000)
            if step % 300 == 0:
                run.log_eval(step, report)
        run.log_metrics(1000, loss=math.nan)
    return run.dir


def test_run_directory_contents(tmp_path, report):
    printed = []
    run_dir = write_run(tmp_path, report, printed=printed)
    assert {path.name for path in run_dir.iterdir()} >= {
        "config.json", "metrics.jsonl", "evals.jsonl", "warnings.jsonl", "summary.json", "checkpoints",
    }
    config = json.loads((run_dir / "config.json").read_text())
    assert config["config"]["learning_rate"] == 0.001 and "commit" in config["git"]

    run = load_run(run_dir)
    assert len(run["metrics"]) == 11 and len(run["evals"]) == 4
    assert run["warnings"][0]["message"] == "metric loss is nan"
    assert run_status(run) == "finished" and run["summary"]["final_step"] == 1000

    # One console line per call, plus the start and end lines
    assert len(printed) == 1 + 11 + 4 + 1 + 1
    assert any(line.startswith("WARN step 1,000: metric loss is nan") for line in printed)


def test_summarize_is_short_and_informative(tmp_path, report):
    lines = summarize(load_run(write_run(tmp_path, report)))
    text = "\n".join(lines)
    assert len(lines) <= 30
    assert "finished" in lines[0]
    assert "learning_rate=0.001 (default 0.0003)" in text and "opponents=" not in text  # only non-defaults
    assert "Metrics (11 records, 5 shown)" in text
    assert "Latest eval (step 900" in text and "maxraise" in text
    assert "Warnings: 1" in text


def test_crashed_and_died_runs(tmp_path):
    with pytest.raises(RuntimeError):
        with RunLogger("crash", {}, root=tmp_path, print_fn=lambda line: None) as run:
            run.log_metrics(5, loss=1.0)
            raise RuntimeError("boom")
    crashed = load_run(run.dir)
    assert run_status(crashed) == "crashed" and "boom" in summarize(crashed)[1]

    died = RunLogger("died", {}, root=tmp_path, print_fn=lambda line: None)
    process = subprocess.Popen(["true"])
    process.wait()
    config = json.loads((died.dir / "config.json").read_text())
    config["pid"] = process.pid
    (died.dir / "config.json").write_text(json.dumps(config))
    assert run_status(load_run(died.dir)) == "died"


def test_compare_groups_seeds(tmp_path, report):
    run_dirs = [write_run(tmp_path, report, seed=seed) for seed in [0, 1]]
    run_dirs.append(write_run(tmp_path, report, seed=0, name="other"))
    lines = compare([load_run(path) for path in run_dirs])
    assert len(lines) == 1 + 3 + 1  # header, one row per run, one seed group
    assert lines[-1].startswith("mean ± sd of 2 seeds (mc_test)")

    listing = list_runs(tmp_path)
    assert len(listing) == 3 and all("finished" in line for line in listing)


def test_plot(tmp_path, report):
    path = plot(load_run(write_run(tmp_path, report)))
    assert path.exists() and path.stat().st_size > 1000
