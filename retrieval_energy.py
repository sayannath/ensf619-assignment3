"""Estimate the energy of one retrieval. A single search is too short for CodeCarbon to measure, so run many."""
import csv

from transformers import AutoTokenizer

from config import CONFIGS, LLM_ID, PAPERS_DIR, RESULTS_DIR
from documents import load_papers
from emissions import tracked
from run_experiment import build_retriever, load_questions

ROUNDS = 200  # each round searches all 10 questions


def search_many(retriever, questions, k):
    for _ in range(ROUNDS):
        for question in questions:
            retriever.search(question["question"], k)


def main():
    questions = load_questions()
    papers = load_papers(PAPERS_DIR)
    tokenizer = AutoTokenizer.from_pretrained(LLM_ID)
    searches = ROUNDS * len(questions)

    with open(RESULTS_DIR / "retrieval_energy.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "searches", "ms_per_search", "energy_kwh_per_search", "co2_kg_per_search"])
        for name, settings in CONFIGS.items():
            retriever, _ = build_retriever(name, settings, papers, tokenizer)
            retriever.search(questions[0]["question"], settings["k"])  # warm-up, not recorded
            _, emissions = tracked(f"{name}-retrieval", search_many, retriever, questions, settings["k"])
            row = [name, searches, emissions.duration * 1000 / searches, emissions.energy_consumed / searches,
                   emissions.emissions / searches]
            writer.writerow(row)
            print(row, flush=True)


if __name__ == "__main__":
    main()
