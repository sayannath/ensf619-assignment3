"""Run the 10 questions through both configurations: 20 rows in total, one per question and configuration."""
import argparse
import csv
from statistics import mean

from config import CONFIGS, LLM_ID, PAPERS_DIR, QUESTIONS_FILE, REPEATS, RESULTS_DIR
from documents import load_papers, make_chunks
from emissions import tracked
from generator import generate_answer, load_llm
from retriever import Retriever

RUN_FIELDS = ["config", "question_id", "t_retriever_ms", "t_generator_ms", "t_total_ms", "t_in", "prompt_tokens",
              "output_tokens", "co2_kg", "energy_kwh", "cpu_kwh", "gpu_kwh", "ram_kwh", "accuracy", "retrieved_chunks",
              "question", "answer"]
INDEX_FIELDS = ["config", "chunks", "truncated_chunks", "longest_chunk", "embedder_limit", "index_ms", "co2_kg",
                "energy_kwh"]


def load_questions():
    with open(QUESTIONS_FILE) as f:
        return list(csv.DictReader(f))


def build_retriever(name, settings, papers, tokenizer):
    """Chunk the papers and embed them. Returns the retriever and a row describing the one-off indexing cost."""
    chunks = make_chunks(papers, tokenizer, settings["chunk_size"], settings["overlap"])
    retriever = Retriever(settings["embedder"], settings["query_prefix"])
    retriever.embed("warm-up")

    def build_repeatedly():
        for _ in range(REPEATS):
            retriever.build_index(chunks)

    _, emissions = tracked(f"{name}-index", build_repeatedly)

    lengths = retriever.chunk_lengths()
    limit = retriever.model.max_seq_length
    row = {"config": name, "chunks": len(chunks), "truncated_chunks": sum(n > limit for n in lengths),
           "longest_chunk": max(lengths), "embedder_limit": limit, "index_ms": emissions.duration * 1000 / REPEATS,
           "co2_kg": emissions.emissions / REPEATS, "energy_kwh": emissions.energy_consumed / REPEATS}
    return retriever, row


def ask(question, retriever, k, model, tokenizer):
    """One RAG query: retrieve the chunks, then generate the answer."""
    chunks, t_retriever_ms = retriever.search(question, k)
    result = generate_answer(model, tokenizer, question, chunks)
    result["t_retriever_ms"] = t_retriever_ms
    result["t_total_ms"] = t_retriever_ms + result["t_generator_ms"]
    # the brief's T_in: query tokens + retrieved chunk tokens. prompt_tokens also counts the instructions.
    result["t_in"] = len(tokenizer.encode(question, add_special_tokens=False)) + sum(c["tokens"] for c in chunks)
    result["retrieved_chunks"] = ";".join(c["id"] for c in chunks)
    return result


def measure(label, question, retriever, k, model, tokenizer):
    """Ask one question REPEATS times inside a single tracker and return the averages for one query."""
    def repeat():
        return [ask(question, retriever, k, model, tokenizer) for _ in range(REPEATS)]

    results, emissions = tracked(label, repeat)
    result = results[0]
    for field in ["t_retriever_ms", "t_generator_ms", "t_total_ms"]:
        result[field] = mean(r[field] for r in results)
    result["co2_kg"] = emissions.emissions / REPEATS
    result["energy_kwh"] = emissions.energy_consumed / REPEATS
    result["cpu_kwh"] = emissions.cpu_energy / REPEATS
    result["gpu_kwh"] = emissions.gpu_energy / REPEATS
    result["ram_kwh"] = emissions.ram_energy / REPEATS
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--questions", type=int, default=10, help="use fewer for a quick test")
    parser.add_argument("--out", default=RESULTS_DIR / "results.csv")
    parser.add_argument("--index-out", default=RESULTS_DIR / "index_costs.csv")
    args = parser.parse_args()

    questions = load_questions()[:args.questions]
    papers = load_papers(PAPERS_DIR)
    model, tokenizer = load_llm(LLM_ID)
    RESULTS_DIR.mkdir(exist_ok=True)

    index_rows = []
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUN_FIELDS)
        writer.writeheader()
        for name in args.configs:
            settings = CONFIGS[name]
            retriever, index_row = build_retriever(name, settings, papers, tokenizer)
            index_rows.append(index_row)
            print(index_row, flush=True)

            ask(questions[0]["question"], retriever, settings["k"], model, tokenizer)  # warm-up, not recorded

            for question in questions:
                result = measure(f"{name}-q{question['id']}", question["question"], retriever, settings["k"], model,
                                 tokenizer)
                writer.writerow({"config": name, "question_id": question["id"], "question": question["question"],
                                 "accuracy": "", **result})
                f.flush()
                print(f"{name} q{question['id']}: retrieve={result['t_retriever_ms']:.1f}ms "
                      f"generate={result['t_generator_ms']:.0f}ms t_in={result['t_in']} "
                      f"out={result['output_tokens']} co2={result['co2_kg']:.2e}kg", flush=True)

    with open(args.index_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(index_rows)


if __name__ == "__main__":
    main()
