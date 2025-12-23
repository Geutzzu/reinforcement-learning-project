import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def load_logs(run_dir: str) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(run_dir, "logs.parquet"))


def load_config(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "config.json")) as f:
        return json.load(f)


def plot_training(run_dir: str, metrics: list[str] = None, save: bool = True):
    df = load_logs(run_dir)
    config = load_config(run_dir)
    
    metrics = metrics or ["loss", "eval_loss", "reward/mean", "kl"]
    metrics = [m for m in metrics if m in df.columns]
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        data = df[["step", metric]].dropna()
        ax.plot(data["step"], data[metric])
        ax.set_xlabel("Step")
        ax.set_ylabel(metric)
        ax.set_title(f"{config['algorithm'].upper()} - {metric}")
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f"{config['algorithm'].upper()}_{config['name']}", y=1.02)
    plt.tight_layout()
    
    if save:
        plt.savefig(os.path.join(run_dir, "training_curves.png"), dpi=150, bbox_inches="tight")
    plt.show()


def plot_comparison(run_dirs: list[str], metric: str = "loss", save_path: str = None):
    plt.figure(figsize=(10, 6))
    
    for run_dir in run_dirs:
        df = load_logs(run_dir)
        config = load_config(run_dir)
        if metric in df.columns:
            data = df[["step", metric]].dropna()
            label = f"{config['algorithm'].upper()}_{config['name']}"
            plt.plot(data["step"], data[metric], label=label)
    
    plt.xlabel("Step")
    plt.ylabel(metric)
    plt.title(f"Comparison: {metric}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_accuracy_comparison(eval_results: dict[str, dict], save_path: str = None):
    """eval_results = {"SFT": {"maze": 0.75}, "GRPO": {"maze": 0.82}, ...}"""
    algorithms = list(eval_results.keys())
    tasks = list(eval_results[algorithms[0]].keys())
    
    x = range(len(tasks))
    width = 0.8 / len(algorithms)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, algo in enumerate(algorithms):
        accuracies = [eval_results[algo][task]["accuracy"] for task in tasks]
        offset = (i - len(algorithms) / 2 + 0.5) * width
        bars = ax.bar([xi + offset for xi in x], accuracies, width, label=algo)
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{acc:.1%}", ha="center", va="bottom", fontsize=9)
    
    ax.set_xlabel("Task")
    ax.set_ylabel("Accuracy")
    ax.set_title("Algorithm Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def find_runs(results_dir: str = "results", name: str = None, algorithm: str = None) -> list[str]:
    runs = []
    for name_dir in Path(results_dir).iterdir():
        if not name_dir.is_dir():
            continue
        if name and name_dir.name != name:
            continue
        for run_dir in name_dir.iterdir():
            if not run_dir.is_dir():
                continue
            if algorithm and not run_dir.name.startswith(algorithm):
                continue
            if (run_dir / "logs.parquet").exists():
                runs.append(str(run_dir))
    return sorted(runs)
