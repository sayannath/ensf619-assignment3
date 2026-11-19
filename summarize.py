"""Print the report's tables from the raw CSV files and save the main one to results/summary.csv."""
import csv
from collections import defaultdict
from statistics import mean

import numpy as np

from config import RESULTS_DIR

AVERAGED = ["t_retriever_ms", "t_generator_ms", "t_total_ms", "t_in", "prompt_tokens", "output_tokens", "co2_kg",
            "energy_kwh", "cpu_kwh", "gpu_kwh", "ram_kwh"]
FIELDS = ["config", "runs"] + AVERAGED + ["passes", "accuracy_pct", "passes_per_second", "passes_per_gram_co2"]


def read_csv(name):
    with open(RESULTS_DIR / name) as f:
        return list(csv.DictReader(f))


def average_by_config(runs):
    groups = defaultdict(list)
    for run in runs:
        groups[run["config"]].append(run)

    rows = []
    for name, group in groups.items():
        row = {"config": name, "runs": len(group)}
        for field in AVERAGED:
            row[field] = mean(float(run[field]) for run in group)
        row["passes"] = sum(run["accuracy"] == "P" for run in group)
        row["accuracy_pct"] = 100 * row["passes"] / len(group)
        row["passes_per_second"] = row["passes"] / (row["t_total_ms"] * len(group) / 1000)
        row["passes_per_gram_co2"] = row["passes"] / (row["co2_kg"] * len(group) * 1000)
        rows.append(row)
    return rows


def print_main_table(rows):
    print("| Config | T_retriever (ms) | T_generator (ms) | T_total (ms) | T_in | Output tokens | Energy (J) | "
          "CO2e per query (kg) | Accuracy | Passes per second | Passes per g CO2e |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['config']} | {r['t_retriever_ms']:.2f} | {r['t_generator_ms']:.0f} | {r['t_total_ms']:.0f} | "
              f"{r['t_in']:.0f} | {r['output_tokens']:.1f} | {r['energy_kwh'] * 3.6e6:.1f} | {r['co2_kg']:.2e} | "
              f"{r['passes']}/{r['runs']} | {r['passes_per_second']:.3f} | {r['passes_per_gram_co2']:.0f} |")


def print_timing_table(rows):
    print("| Config | T_retriever (ms) | T_generator (ms) | T_total (ms) | Energy (J) | CO2e per query (kg) |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['config']} | {r['t_retriever_ms']:.2f} | {r['t_generator_ms']:.0f} | {r['t_total_ms']:.0f} | "
              f"{r['energy_kwh'] * 3.6e6:.1f} | {r['co2_kg']:.2e} |")


def print_k_sweep(runs):
    groups = defaultdict(list)
    for run in runs:
        groups[(run["config"], int(run["k"]))].append(run)

    print("| Config | k | T_in | Output tokens | T_generator (ms) | Energy (J) | CO2e per query (kg) |")
    print("|---|---|---|---|---|---|---|")
    for (name, k), group in groups.items():
        values = {f: mean(float(run[f]) for run in group) for f in
                  ["t_in", "output_tokens", "t_generator_ms", "energy_kwh", "co2_kg"]}
        print(f"| {name} | {k} | {values['t_in']:.0f} | {values['output_tokens']:.1f} | "
              f"{values['t_generator_ms']:.0f} | {values['energy_kwh'] * 3.6e6:.1f} | {values['co2_kg']:.2e} |")

    # Generation time also depends on answer length, so fit both token counts together.
    inputs = np.array([float(run["prompt_tokens"]) for run in runs])
    outputs = np.array([float(run["output_tokens"]) for run in runs])
    times = np.array([float(run["t_generator_ms"]) for run in runs])
    joules = np.array([float(run["energy_kwh"]) * 3.6e6 for run in runs])
    design = np.column_stack([np.ones(len(runs)), inputs, outputs])
    for label, target, unit in [("T_generator", times, "ms"), ("Energy", joules, "J")]:
        fit, *_ = np.linalg.lstsq(design, target, rcond=None)
        r_squared = 1 - np.sum((target - design @ fit) ** 2) / np.sum((target - target.mean()) ** 2)
        print(f"{label} = {fit[0]:.1f} {unit} + {fit[1] * 1000:.1f} {unit} per 1000 input tokens "
              f"+ {fit[2]:.2f} {unit} per output token (R^2 = {r_squared:.3f})")


def print_shared_passes(runs):
    """Compare only the questions both configurations answered correctly, so a short "I don't know" can't look cheap."""
    passes = defaultdict(set)
    for run in runs:
        if run["accuracy"] == "P":
            passes[run["config"]].add(run["question_id"])
    shared = set.intersection(*passes.values()) if len(passes) == 2 else set()
    print(f"Questions passed by both: {sorted(shared, key=int)}")
    if shared:
        print_main_table(average_by_config([run for run in runs if run["question_id"] in shared]))


def print_fact_check(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["config"], row["k"])].append(row)

    print("| Config | k | Facts retrieved | Facts in answer |")
    print("|---|---|---|---|")
    for (name, k), group in groups.items():
        retrieved = sum(row["facts_retrieved"] == "True" for row in group)
        answered = sum(row["facts_in_answer"] == "True" for row in group)
        print(f"| {name} | {k} | {retrieved}/{len(group)} | {answered}/{len(group)} |")


def print_one_off_costs():
    for row in read_csv("index_costs.csv"):
        print(f"Index {row['config']}: {row['chunks']} chunks, {row['truncated_chunks']} truncated, "
              f"longest {row['longest_chunk']} of {row['embedder_limit']} tokens, {float(row['index_ms']):.0f} ms, "
              f"{float(row['energy_kwh']) * 3.6e6:.1f} J, {float(row['co2_kg']):.2e} kg")
    for row in read_csv("retrieval_energy.csv"):
        print(f"Retrieval {row['config']}: {float(row['ms_per_search']):.2f} ms, "
              f"{float(row['energy_kwh_per_search']) * 3.6e6:.3f} J, {float(row['co2_kg_per_search']):.2e} kg")


def main():
    results = read_csv("results.csv")
    if any(run["accuracy"] == "" for run in results):
        print("Warning: some runs are not graded yet, so the accuracy columns are not final.\n")

    rows = average_by_config(results)
    with open(RESULTS_DIR / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print("Main comparison")
    print_main_table(rows)
    print()
    print_shared_passes(results)
    print("\nRepeat run in reversed order (timing and energy check, not graded)")
    print_timing_table(average_by_config(read_csv("check_reversed.csv")))
    print("\nk sweep (not graded)")
    print_k_sweep(read_csv("k_sweep.csv"))
    print("\nKeyword check of the k sweep (not a hand grade)")
    print_fact_check(read_csv("fact_check.csv"))
    print("\nIndexing and retrieval costs")
    print_one_off_costs()


if __name__ == "__main__":
    main()
