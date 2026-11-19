"""Change only k, with the embedding model and chunk size fixed, to isolate its effect on generation time."""
import csv

from config import CONFIGS, LLM_ID, PAPERS_DIR, RESULTS_DIR
from documents import load_papers
from generator import load_llm
from run_experiment import ask, build_retriever, load_questions, measure

K_VALUES = [1, 3, 5, 8]
FIELDS = ["config", "k", "question_id", "t_retriever_ms", "t_generator_ms", "t_in", "prompt_tokens", "output_tokens",
          "co2_kg", "energy_kwh", "retrieved_chunks", "answer"]


def main():
    questions = load_questions()
    papers = load_papers(PAPERS_DIR)
    model, tokenizer = load_llm(LLM_ID)

    with open(RESULTS_DIR / "k_sweep.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for name, settings in CONFIGS.items():
            retriever, _ = build_retriever(name, settings, papers, tokenizer)
            ask(questions[0]["question"], retriever, 1, model, tokenizer)  # warm-up, not recorded
            for k in K_VALUES:
                for question in questions:
                    result = measure(f"{name}-k{k}-q{question['id']}", question["question"], retriever, k, model,
                                     tokenizer)
                    writer.writerow({"config": name, "k": k, "question_id": question["id"], **result})
                f.flush()
                print(f"{name} k={k} done", flush=True)


if __name__ == "__main__":
    main()
