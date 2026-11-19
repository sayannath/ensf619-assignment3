# Assignment 3: Carbon-Efficient RAG Architecture Design

Compares two RAG configurations on 10 factual questions over three technical papers. The findings are in [report.md](report.md).

| Setting | Baseline | Optimized |
|---|---|---|
| Embedding model | `all-MiniLM-L6-v2` | `bge-small-en-v1.5` |
| Chunk size / overlap | 256 / 50 tokens | 512 / 50 tokens |
| k | 8 | 3 |
| Generator | `Qwen2.5-7B-Instruct` | Same |

## Layout

```
config.py              settings for both configurations
documents.py           load the PDFs and split them into chunks
retriever.py           embedding model and vector store
generator.py           load the LLM and generate an answer
emissions.py           CodeCarbon tracker
run_experiment.py      the main 20 runs
k_sweep.py             change only k
check_facts.py         keyword check of retrieval and answers in the k sweep
retrieval_energy.py    energy of one retrieval, averaged over 2,000 searches
make_grading_sheet.py  shuffled sheet for blind grading
apply_grades.py        copy the grades into results.csv
summarize.py           print the report's tables
data/                  the papers, questions and expected answers
results/               raw and summarized results
logs/                  output of the final run
```

## Setup

Requires a CUDA GPU.

```bash
pip install -r requirements.txt
```

## Run everything

```bash
bash run_all.sh
```

This overwrites the files in `results/`, including the grading sheet. To grade afterwards, fill the `accuracy` column of `results/grading_sheet.csv` with `P` or `F`, then run:

```bash
python apply_grades.py
python summarize.py
```

## Measurements

Each row in `results/results.csv` is the per-query mean of 10 executions inside one CodeCarbon tracker, because a single query is shorter than the tracker's 1 s sampling interval. Emissions use the Alberta grid.

`T_in` is query tokens plus retrieved chunk tokens. `prompt_tokens` also counts the instructions and chat formatting.
