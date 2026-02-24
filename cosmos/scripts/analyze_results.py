#!/usr/bin/env python3
"""
COSMOS - Results Analysis Script

Analyzes experiment results and generates comparison plots.

Usage:
    python scripts/analyze_results.py experiments/20240101_120000
"""

import os
import sys
import json
import glob
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def load_metrics(results_dir: str) -> dict:
    """Load metrics from all experiments."""
    metrics = defaultdict(list)

    # Load from JSON files
    for json_file in glob.glob(f"{results_dir}/**/metrics.json", recursive=True):
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Extract experiment name from path
        exp_name = Path(json_file).parent.name
        metrics[exp_name] = data

    return dict(metrics)


def parse_experiment_name(name: str) -> dict:
    """Parse experiment name to extract components."""
    parts = name.split('_')

    result = {
        'env': parts[0] if len(parts) > 0 else 'unknown',
        'algo': parts[1] if len(parts) > 1 else 'unknown',
        'safety': parts[2] if len(parts) > 2 else 'none',
        'seed': int(parts[3].replace('seed', '')) if len(parts) > 3 else 0,
    }

    return result


def aggregate_seeds(metrics: dict) -> dict:
    """Aggregate results across seeds."""
    aggregated = defaultdict(lambda: defaultdict(list))

    for exp_name, data in metrics.items():
        info = parse_experiment_name(exp_name)
        key = f"{info['env']}_{info['algo']}_{info['safety']}"

        for metric_name, values in data.items():
            if isinstance(values, list):
                aggregated[key][metric_name].append(values)

    # Compute mean and std
    results = {}
    for key, metric_data in aggregated.items():
        results[key] = {}
        for metric_name, seed_values in metric_data.items():
            # Align lengths
            min_len = min(len(v) for v in seed_values)
            aligned = [v[:min_len] for v in seed_values]

            arr = np.array(aligned)
            results[key][metric_name] = {
                'mean': arr.mean(axis=0).tolist(),
                'std': arr.std(axis=0).tolist(),
                'min': arr.min(axis=0).tolist(),
                'max': arr.max(axis=0).tolist(),
            }

    return results


def plot_learning_curves(results: dict, output_dir: str):
    """Plot learning curves for different methods."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics_to_plot = ['reward', 'cost', 'formation_error', 'collisions']
    titles = ['Episode Reward', 'Episode Cost', 'Formation Error', 'Collisions']

    colors = plt.cm.tab10.colors

    for ax, metric, title in zip(axes.flatten(), metrics_to_plot, titles):
        color_idx = 0
        for exp_name, data in results.items():
            if metric not in data:
                continue

            mean = np.array(data[metric]['mean'])
            std = np.array(data[metric]['std'])
            episodes = np.arange(len(mean))

            # Parse label
            parts = exp_name.split('_')
            label = f"{parts[1]}-{parts[2]}" if len(parts) >= 3 else exp_name

            ax.plot(episodes, mean, label=label, color=colors[color_idx % len(colors)])
            ax.fill_between(episodes, mean - std, mean + std, alpha=0.2, color=colors[color_idx % len(colors)])
            color_idx += 1

        ax.set_xlabel('Episode')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/learning_curves.png", dpi=150)
    plt.savefig(f"{output_dir}/learning_curves.pdf")
    print(f"Saved: {output_dir}/learning_curves.png")
    plt.close()


def plot_safety_comparison(results: dict, output_dir: str):
    """Plot safety method comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Group by safety method
    safety_methods = defaultdict(list)
    for exp_name, data in results.items():
        parts = exp_name.split('_')
        if len(parts) >= 3:
            safety = parts[2]
            safety_methods[safety].append({
                'name': exp_name,
                'data': data
            })

    metrics = ['reward', 'cost', 'collisions']
    titles = ['Final Reward ↑', 'Total Cost ↓', 'Total Collisions ↓']

    for ax, metric, title in zip(axes, metrics, titles):
        labels = []
        values = []
        errors = []

        for safety, experiments in safety_methods.items():
            for exp in experiments:
                if metric in exp['data']:
                    mean_vals = exp['data'][metric]['mean']
                    std_vals = exp['data'][metric]['std']

                    # Use last 10% as final performance
                    n = max(1, len(mean_vals) // 10)
                    final_mean = np.mean(mean_vals[-n:])
                    final_std = np.mean(std_vals[-n:])

                    labels.append(safety)
                    values.append(final_mean)
                    errors.append(final_std)

        x = np.arange(len(labels))
        ax.bar(x, values, yerr=errors, capsize=5, color=['green', 'blue', 'red'][:len(labels)])
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(title)
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/safety_comparison.png", dpi=150)
    plt.savefig(f"{output_dir}/safety_comparison.pdf")
    print(f"Saved: {output_dir}/safety_comparison.png")
    plt.close()


def generate_latex_table(results: dict, output_dir: str):
    """Generate LaTeX table for paper."""
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Performance Comparison}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Reward $\uparrow$ & Cost $\downarrow$ & Formation Error $\downarrow$ & Collisions $\downarrow$ \\")
    lines.append(r"\midrule")

    for exp_name, data in results.items():
        parts = exp_name.split('_')
        method = f"{parts[1]}-{parts[2]}" if len(parts) >= 3 else exp_name

        row = [method]
        for metric in ['reward', 'cost', 'formation_error', 'collisions']:
            if metric in data:
                mean = np.mean(data[metric]['mean'][-10:])
                std = np.mean(data[metric]['std'][-10:])
                row.append(f"${mean:.2f} \\pm {std:.2f}$")
            else:
                row.append("-")

        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    table_file = f"{output_dir}/results_table.tex"
    with open(table_file, 'w') as f:
        f.write("\n".join(lines))
    print(f"Saved: {table_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze COSMOS experiment results')
    parser.add_argument('results_dir', type=str, help='Directory containing experiment results')
    parser.add_argument('--output', type=str, default=None, help='Output directory for plots')
    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir = args.output or f"{results_dir}/analysis"

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading results from: {results_dir}")

    # Load and process metrics
    metrics = load_metrics(results_dir)
    print(f"Found {len(metrics)} experiments")

    if len(metrics) == 0:
        print("No metrics found. Make sure experiments have completed.")
        return

    # Aggregate across seeds
    results = aggregate_seeds(metrics)
    print(f"Aggregated into {len(results)} unique configurations")

    # Generate plots
    print("\nGenerating plots...")
    plot_learning_curves(results, output_dir)
    plot_safety_comparison(results, output_dir)

    # Generate LaTeX table
    print("\nGenerating LaTeX table...")
    generate_latex_table(results, output_dir)

    # Save aggregated results
    with open(f"{output_dir}/aggregated_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {output_dir}/aggregated_results.json")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
