"""Keyword check of the k sweep: did the retrieved chunks, and then the answer, contain the facts asked for?

This separates retrieval misses from generation errors. It does not replace the hand grades in results.csv.
"""
import csv

from transformers import AutoTokenizer

from config import CONFIGS, LLM_ID, PAPERS_DIR, RESULTS_DIR
from documents import load_papers, make_chunks

FACTS = {
    "1": ["1.58"],
    "2": ["626,155"],
    "3": ["4789", "9998"],
    "4": ["example", "dataset", "hyperparameter"],
    "5": ["90%", "80%", "75%"],
    "6": ["350 million"],
    "7": ["floating point"],
    "8": ["100-word", "21m"],
    "9": ["same", "different"],
    "10": ["71%", "90%"],
}


def has_facts(text, facts):
    return all(fact in text.lower() for fact in facts)


def main():
    papers = load_papers(PAPERS_DIR)
    tokenizer = AutoTokenizer.from_pretrained(LLM_ID)
    chunk_text = {}
    for name, settings in CONFIGS.items():
        chunks = make_chunks(papers, tokenizer, settings["chunk_size"], settings["overlap"])
        chunk_text[name] = {c["id"]: c["text"] for c in chunks}

    with open(RESULTS_DIR / "k_sweep.csv") as f:
        runs = list(csv.DictReader(f))

    with open(RESULTS_DIR / "fact_check.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "k", "question_id", "facts_retrieved", "facts_in_answer"])
        for run in runs:
            ids = run["retrieved_chunks"].split(";")
            context = " ".join(chunk_text[run["config"]][chunk_id] for chunk_id in ids)
            facts = FACTS[run["question_id"]]
            writer.writerow([run["config"], run["k"], run["question_id"], has_facts(context, facts),
                             has_facts(run["answer"], facts)])


if __name__ == "__main__":
    main()
