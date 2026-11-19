from pathlib import Path

ROOT = Path(__file__).parent
PAPERS_DIR = ROOT / "data" / "papers"
QUESTIONS_FILE = ROOT / "data" / "questions.csv"
RESULTS_DIR = ROOT / "results"

LLM_ID = "Qwen/Qwen2.5-7B-Instruct"
MAX_NEW_TOKENS = 128

# One query is shorter than CodeCarbon's 1 s sampling interval, so each one runs this many times per measurement.
REPEATS = 10

CONFIGS = {
    "baseline": {
        "embedder": "sentence-transformers/all-MiniLM-L6-v2",
        "query_prefix": "",
        "chunk_size": 256,
        "overlap": 50,
        "k": 8,
    },
    "optimized": {
        "embedder": "BAAI/bge-small-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",  # recommended by BGE
        "chunk_size": 512,
        "overlap": 50,
        "k": 3,
    },
}
