"""
Run directories for training runs, and a CLI that summarizes and compares them in a few lines.

A run directory (runs/<YYYY-MM-DD_HHMMSS>_<name>/) holds:
    config.json     settings, their defaults, git commit, start time, pid, command line
    metrics.jsonl   one JSON object per logging interval: {"step", "wall_s", <metrics>...}
    evals.jsonl     one poker.benchmark report per evaluation, plus "step" and "wall_s"
    warnings.jsonl  one {"step", "message"} per warning (non-finite metrics are flagged automatically)
    summary.json    written at the end: status ("finished" or "crashed"), wall time, final step
    checkpoints/    model files (RunLogger.checkpoint_path)

Trainers write with RunLogger, which also prints one short console line per call. To look at
runs, use the CLI rather than reading the files:

    uv run python -m poker.runs list
    uv run python -m poker.runs summarize runs/2026-10-03_120000_mc_callstation
    uv run python -m poker.runs compare runs/*_mc_*         # groups runs that differ only by seed
    uv run python -m poker.runs plot runs/2026-10-03_120000_mc_callstation   # writes curves.png
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np

RUNS_ROOT = "runs"


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (set, tuple)):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def _append_jsonl(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record, default=_json_default) + "\n")


def _read_jsonl(path):
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def git_info():
    def git(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True, timeout=10).stdout.strip()

    try:
        return {"commit": git("rev-parse", "--short", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def format_duration(seconds):
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


def format_number(value):
    if value is None:
        return "-"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, int) or (isinstance(value, float) and value.is_integer() and abs(value) >= 1000):
        return f"{int(value):,}"
    if not math.isfinite(value):
        return str(value)
    if value != 0 and (abs(value) < 1e-3 or abs(value) >= 1e5):
        return f"{value:.3g}"
    return f"{value:.4g}"


class RunLogger:
    """
    Writes a run directory (see the module docstring) and prints one short line per call.

    Use it as a context manager so a crash still writes summary.json with status "crashed":

        with RunLogger("mc_callstation", config, defaults=DEFAULTS) as run:
            run.log_metrics(step, loss=..., q_abs=...)
            run.log_eval(step, run_benchmark(...))
            run.save(...) / run.checkpoint_path(step)
    """

    def __init__(self, name, config, defaults=None, root=RUNS_ROOT, console_keys=None, print_fn=None):
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.dir = Path(root) / f"{stamp}_{name}"
        suffix = 1
        while self.dir.exists():
            suffix += 1
            self.dir = Path(root) / f"{stamp}_{name}_{suffix}"
        (self.dir / "checkpoints").mkdir(parents=True)

        self.console_keys = console_keys
        # Note: flush each line, so a console redirected to a file (e.g. a background run) stays current
        self.print = print_fn or partial(print, flush=True)
        self.start_time = time.time()
        self.last_step = None
        self.n_warnings = 0

        with open(self.dir / "config.json", "w") as f:
            json.dump({
                "name": name,
                "config": config,
                "defaults": defaults or {},
                "git": git_info(),
                "started": datetime.now().isoformat(timespec="seconds"),
                "pid": os.getpid(),
                "command": sys.argv,
            }, f, indent=1, default=_json_default)
        self.print(f"Run directory: {self.dir}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if not (self.dir / "summary.json").exists():
            if exc_type is None:
                self.finish()
            else:
                self.finish(status="crashed", error=f"{exc_type.__name__}: {exc}")
        return False

    def wall_s(self):
        return round(time.time() - self.start_time, 1)

    def log_metrics(self, step, **values):
        self.last_step = step
        _append_jsonl(self.dir / "metrics.jsonl", {"step": step, "wall_s": self.wall_s(), **values})

        for key, value in values.items():
            if isinstance(value, float) and not math.isfinite(value):
                self.warn(f"metric {key} is {value}", step=step)

        keys = self.console_keys if self.console_keys is not None else list(values)
        fields = " | ".join(f"{key} {format_number(values[key])}" for key in keys if key in values)
        self.print(f"step {format_number(step)} | {format_duration(self.wall_s())} | {fields}")

    def log_eval(self, step, report):
        """Record a poker.benchmark report (run_benchmark) and print one line of chips per deal."""
        self.last_step = step
        _append_jsonl(self.dir / "evals.jsonl", {"step": step, "wall_s": self.wall_s(), **report})
        chips = " ".join(f"{name} {result['chips_per_deal']:+.2f}" for name, result in report["opponents"].items())
        m1 = report.get("m1", {})
        self.print(f"eval step {format_number(step)} | {chips} | M1 {m1.get('passed', '?')}/{m1.get('checked', '?')}")

    def warn(self, message, step=None):
        self.n_warnings += 1
        step = self.last_step if step is None else step
        _append_jsonl(self.dir / "warnings.jsonl", {"step": step, "wall_s": self.wall_s(), "message": message})
        self.print(f"WARN step {format_number(step)}: {message}")

    def checkpoint_path(self, step, suffix=".weights.h5"):
        return self.dir / "checkpoints" / f"step_{step:010d}{suffix}"

    def finish(self, status="finished", **extra):
        summary = {"status": status, "wall_s": self.wall_s(), "final_step": self.last_step,
                   "n_warnings": self.n_warnings, **extra}
        with open(self.dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=1, default=_json_default)
        self.print(f"Run {status} after {format_duration(summary['wall_s'])}: {self.dir}")


# ---------------------------------------------------------------------------------------------
# Reading runs


def load_run(path):
    path = Path(path)
    if not (path / "config.json").exists():
        raise FileNotFoundError(f"{path} is not a run directory (no config.json)")
    with open(path / "config.json") as f:
        config = json.load(f)
    summary = None
    if (path / "summary.json").exists():
        with open(path / "summary.json") as f:
            summary = json.load(f)
    return {
        "dir": path,
        "name": path.name,
        "config": config,
        "metrics": _read_jsonl(path / "metrics.jsonl"),
        "evals": _read_jsonl(path / "evals.jsonl"),
        "warnings": _read_jsonl(path / "warnings.jsonl"),
        "summary": summary,
    }


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except (OSError, TypeError):
        return False
    return True


def run_status(run):
    if run["summary"] is not None:
        return run["summary"]["status"]
    return "running" if _pid_alive(run["config"].get("pid")) else "died"


def _last(records, key):
    return records[-1].get(key) if records else None


def run_progress(run):
    """(final step, wall seconds) from the summary, or from the latest records if still running."""
    if run["summary"] is not None:
        return run["summary"].get("final_step"), run["summary"].get("wall_s")
    records = [record for record in run["metrics"] + run["evals"] if "step" in record]
    if not records:
        return None, None
    latest = max(records, key=lambda record: record.get("wall_s", 0))
    return latest.get("step"), latest.get("wall_s")


def panel_score(report):
    """Mean chips per deal over the opponents in a benchmark report (one number per eval)."""
    values = [result["chips_per_deal"] for result in report["opponents"].values()]
    return float(np.mean(values)) if values else None


def config_differences(config_record):
    config, defaults = config_record.get("config", {}), config_record.get("defaults", {})
    if not defaults:
        return [f"{key}={value}" for key, value in config.items()]
    return [f"{key}={config[key]} (default {defaults[key]})" if key in defaults else f"{key}={config[key]}"
            for key in config if key not in defaults or config[key] != defaults[key]]


def _evenly_spaced(records, n):
    if len(records) <= n:
        return records
    indices = np.unique(np.linspace(0, len(records) - 1, n).round().astype(int))
    return [records[i] for i in indices]


def summarize(run, n_points=5, max_metric_columns=8, include_fingerprint=True):
    """A short text summary of a run (about 25 lines)."""
    from poker.benchmark import format_report

    config_record = run["config"]
    step, wall_s = run_progress(run)
    git = config_record.get("git", {})
    git_text = f"git {git.get('commit')}{'+dirty' if git.get('dirty') else ''}" if git.get("commit") else ""
    lines = [" | ".join(part for part in [
        f"Run {run['name']}: {run_status(run)}",
        format_duration(wall_s) if wall_s is not None else None,
        f"step {format_number(step)}" if step is not None else None,
        git_text,
    ] if part)]
    if run["summary"] and run["summary"].get("error"):
        lines.append(f"Error: {run['summary']['error']}")

    differences = config_differences(config_record)
    if differences:
        text = ", ".join(differences)
        lines.append(f"Config{' (non-default)' if config_record.get('defaults') else ''}: "
                     f"{text if len(text) <= 300 else text[:297] + '...'}")

    metrics = [record for record in run["metrics"] if "step" in record]
    if metrics:
        keys = []
        for record in metrics:
            for key, value in record.items():
                if key not in ("step", "wall_s") and key not in keys and isinstance(value, (int, float)):
                    keys.append(key)
        keys = keys[:max_metric_columns]
        rows = [["step", "wall"] + keys] + [
            [format_number(record["step"]), format_duration(record.get("wall_s", 0))]
            + [format_number(record.get(key)) for key in keys]
            for record in _evenly_spaced(metrics, n_points)
        ]
        widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
        lines.append(f"Metrics ({len(metrics)} records, {min(n_points, len(metrics))} shown):")
        lines.extend("  " + "  ".join(cell.rjust(width) for cell, width in zip(row, widths)) for row in rows)

    if run["evals"]:
        latest = run["evals"][-1]
        report = dict(latest) if include_fingerprint else {k: v for k, v in latest.items() if k != "fingerprint"}
        lines.append(f"Latest eval (step {format_number(latest['step'])}, {latest['n_decks']} decks):")
        lines.extend("  " + line for line in format_report(report, title=False))
        if len(run["evals"]) > 1:
            best = max(run["evals"], key=lambda record: (record["m1"]["passed"], panel_score(record)))
            trend = " -> ".join(f"{panel_score(record):+.2f}" for record in _evenly_spaced(run["evals"], n_points))
            lines.append(f"Panel mean chips/deal over {len(run['evals'])} evals: {trend}; "
                         f"best at step {format_number(best['step'])} "
                         f"(M1 {best['m1']['passed']}/{best['m1']['checked']}, {panel_score(best):+.2f})")

    if run["warnings"]:
        first, last = run["warnings"][0], run["warnings"][-1]
        text = f"Warnings: {len(run['warnings'])}; first at step {format_number(first['step'])}: {first['message']}"
        if len(run["warnings"]) > 1:
            text += f"; last at step {format_number(last['step'])}: {last['message']}"
        lines.append(text)
    return lines


def _seed_group_key(run):
    """Runs with the same name and config apart from the seed are repeats of one experiment."""
    config = {key: value for key, value in run["config"].get("config", {}).items() if key != "seed"}
    return json.dumps([run["config"].get("name"), config], sort_keys=True, default=_json_default)


def compare(runs):
    """One row per run (latest eval), plus mean ± sd rows for runs that differ only by seed."""
    from poker.benchmark import DEFAULT_PANEL, display_name

    opponents = []
    for run in runs:
        for name in (run["evals"][-1]["opponents"] if run["evals"] else {}):
            if name not in opponents:
                opponents.append(name)
    opponents.sort(key=lambda name: DEFAULT_PANEL.index(name) if name in DEFAULT_PANEL else len(DEFAULT_PANEL))
    labels = [display_name(name)[:11] for name in opponents]

    header = ["run", "status", "step", "wall"] + labels + ["M1", "panel"]
    rows = []
    for run in runs:
        step, wall_s = run_progress(run)
        row = [run["name"], run_status(run), format_number(step), format_duration(wall_s) if wall_s is not None else "-"]
        if run["evals"]:
            latest = run["evals"][-1]
            row += [f"{latest['opponents'][name]['chips_per_deal']:+.2f}" if name in latest["opponents"] else "-"
                    for name in opponents]
            row += [f"{latest['m1']['passed']}/{latest['m1']['checked']}", f"{panel_score(latest):+.2f}"]
        else:
            row += ["-"] * (len(opponents) + 2)
        rows.append(row)

    groups = {}
    for run in runs:
        if run["evals"]:
            groups.setdefault(_seed_group_key(run), []).append(run)
    for group in groups.values():
        if len(group) < 2:
            continue
        reports = [run["evals"][-1] for run in group]
        row = [f"mean ± sd of {len(group)} seeds ({group[0]['config'].get('name', '')})", "", "", ""]
        for name in opponents:
            values = [report["opponents"][name]["chips_per_deal"] for report in reports if name in report["opponents"]]
            row.append(f"{np.mean(values):+.2f}±{np.std(values, ddof=1):.2f}" if len(values) > 1 else "-")
        scores = [panel_score(report) for report in reports]
        row += ["", f"{np.mean(scores):+.2f}±{np.std(scores, ddof=1):.2f}"]
        rows.append(row)

    table = [header] + rows
    widths = [max(len(row[i]) for row in table) for i in range(len(header))]
    return ["  ".join(cell.ljust(width) if i == 0 else cell.rjust(width) for i, (cell, width) in
                      enumerate(zip(row, widths))) for row in table]


def list_runs(root=RUNS_ROOT, limit=15):
    root = Path(root)
    paths = sorted((path for path in root.iterdir() if (path / "config.json").exists()), reverse=True) \
        if root.exists() else []
    lines = []
    for path in paths[:limit]:
        run = load_run(path)
        step, wall_s = run_progress(run)
        text = f"{run['name']}  {run_status(run)}  step {format_number(step)}"
        if wall_s is not None:
            text += f"  {format_duration(wall_s)}"
        if run["evals"]:
            latest = run["evals"][-1]
            text += f"  M1 {latest['m1']['passed']}/{latest['m1']['checked']}  panel {panel_score(latest):+.2f}"
        if run["warnings"]:
            text += f"  {len(run['warnings'])} warnings"
        lines.append(text)
    if len(paths) > limit:
        lines.append(f"... and {len(paths) - limit} older runs")
    return lines or [f"No runs in {root}/"]


def plot(run, out=None, max_metrics=6):
    """Learning curves (metrics, and chips per deal per opponent with 95% CIs) as a PNG; returns its path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [record for record in run["metrics"] if "step" in record]
    keys = []
    for record in metrics:
        for key, value in record.items():
            if key not in ("step", "wall_s") and key not in keys and isinstance(value, (int, float)):
                keys.append(key)
    keys = keys[:max_metrics]
    n_panels = len(keys) + (1 if run["evals"] else 0)
    if n_panels == 0:
        raise ValueError(f"{run['dir']} has no metrics or evals to plot")

    n_columns = min(3, n_panels)
    n_rows = math.ceil(n_panels / n_columns)
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(4.5 * n_columns, 3.2 * n_rows), squeeze=False)
    axes = axes.flatten()
    for ax, key in zip(axes, keys):
        points = [(record["step"], record[key]) for record in metrics if isinstance(record.get(key), (int, float))]
        ax.plot(*zip(*points), linewidth=1)
        ax.set_title(key, fontsize=10)
        ax.set_xlabel("step", fontsize=8)
        ax.tick_params(labelsize=8)

    if run["evals"]:
        ax = axes[len(keys)]
        steps = [record["step"] for record in run["evals"]]
        for name in run["evals"][-1]["opponents"]:
            values = np.array([record["opponents"].get(name, {}).get("chips_per_deal", np.nan)
                               for record in run["evals"]])
            cis = np.array([record["opponents"].get(name, {}).get("chips_ci", np.nan) for record in run["evals"]])
            (line,) = ax.plot(steps, values, marker="o", markersize=3, linewidth=1, label=name)
            ax.fill_between(steps, values - cis, values + cis, color=line.get_color(), alpha=0.15)
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_title("chips/deal vs 2x opponent", fontsize=10)
        ax.set_xlabel("step", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7)

    for ax in axes[n_panels:]:
        ax.axis("off")
    fig.suptitle(run["name"], fontsize=10)
    fig.tight_layout()
    out = Path(out) if out else run["dir"] / "curves.png"
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    list_parser = commands.add_parser("list", help="recent runs, one line each")
    list_parser.add_argument("--root", default=RUNS_ROOT)
    list_parser.add_argument("--limit", type=int, default=15)

    summarize_parser = commands.add_parser("summarize", help="a short summary of each run")
    summarize_parser.add_argument("runs", nargs="+")
    summarize_parser.add_argument("--no-fingerprint", action="store_true")

    compare_parser = commands.add_parser("compare", help="one row per run, grouped by seed")
    compare_parser.add_argument("runs", nargs="+")

    plot_parser = commands.add_parser("plot", help="write learning curves to <run>/curves.png")
    plot_parser.add_argument("run")
    plot_parser.add_argument("--out")

    args = parser.parse_args(argv)
    if args.command == "list":
        print("\n".join(list_runs(args.root, args.limit)))
    elif args.command == "summarize":
        for i, path in enumerate(args.runs):
            if i:
                print()
            print("\n".join(summarize(load_run(path), include_fingerprint=not args.no_fingerprint)))
    elif args.command == "compare":
        print("\n".join(compare([load_run(path) for path in args.runs])))
    elif args.command == "plot":
        print(plot(load_run(args.run), args.out))


if __name__ == "__main__":
    main()
