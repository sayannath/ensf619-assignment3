# Assignment 3: Carbon-Efficient RAG Architecture Design

**Course:** ENSF 619 | **Author:** Sayan Nath | **Code repository:** [add repository link]

> **Draft note, delete before submitting:** the Pass/Fail grades below were proposed by an AI assistant from a shuffled sheet that hid the configurations. The brief requires human evaluation, so a person must review `results/grading_sheet.csv`, rerun `apply_grades.py` and `summarize.py`, and update any affected numbers here.

## Summary

The Baseline configuration was the more carbon-efficient choice for this task. It passed 9 of 10 questions against 6 of 10 for Optimized, and used 230 J per correct answer against 316 J. Optimized was 12–13% faster and used 8–15% less energy per query, but most of that saving came from cheap "I don't know" answers on questions it failed. All five failed runs were retrieval misses, so limiting retrieval to three chunks cost more accuracy than it saved in energy.

## a) System and Methodology

### Models and pipeline

| Component | Baseline | Optimized |
|---|---|---|
| Embedding model | `all-MiniLM-L6-v2` | `bge-small-en-v1.5` with its recommended query prefix |
| Chunk size / overlap | 256 / 50 tokens | 512 / 50 tokens |
| k | 8 | 3 |
| Chunks in the index | 101 | 45 |
| Generator | `Qwen2.5-7B-Instruct` (bfloat16) | Same |

- **Generator settings:** greedy decoding and a limit of 128 new tokens. The prompt told the model to answer only from the context and to say "I don't know" otherwise.
- **Chunking:** chunk sizes were counted in the generator's tokens, and chunks never span two papers.
- **Vector store:** an exact cosine-similarity search over a NumPy matrix.
- **Hardware:** one NVIDIA A100-PCIE-40GB ran the generator and the embedding models. The session was limited to one CPU core.

### Knowledge base

The knowledge base is three publicly available arXiv papers (37 PDF pages), included in `data/papers/`:

1. Lewis et al. (2020), *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, arXiv:2005.11401
2. Schwartz et al. (2019), *Green AI*, arXiv:1907.10597
3. Strubell et al. (2019), *Energy and Policy Considerations for Deep Learning in NLP*, arXiv:1906.02243

Text was extracted with `pypdf`. Each paper's text was cut at its reference list, which also removes the appendices of Lewis et al. About 21,000 tokens remain.

### Test questions

The questions and expected answers were fixed before the measured runs.

| # | Question | Expected answer |
|---|---|---|
| 1 | What Power Usage Effectiveness (PUE) coefficient do Strubell et al. use in their energy estimates? | 1.58, the 2018 global average for data centers |
| 2 | According to Strubell et al., how many pounds of CO2e are estimated for training a Transformer with neural architecture search? | 626,155 lbs |
| 3 | In the LISA case study by Strubell et al., how many training jobs were run and what was the total GPU time? | 4789 jobs and 9998 days (27 years) |
| 4 | What three quantities does the Red AI equation in the Green AI paper say the cost of a result grows with? | Cost of processing one example, training dataset size, number of hyperparameter experiments |
| 5 | In the Green AI paper's sample of 60 conference papers, what percentage of ACL, NeurIPS and CVPR papers target accuracy? | 90%, 80% and 75% |
| 6 | According to the Green AI paper, how many parameters does BERT-large contain? | Roughly 350 million |
| 7 | What efficiency measure does the Green AI paper advocate reporting and what does it count? | FPO, the total number of floating point operations required to generate a result |
| 8 | How do Lewis et al. split Wikipedia for the RAG document index and how many documents does this produce? | Disjoint 100-word chunks, giving 21M documents |
| 9 | What is the difference between the RAG-Sequence and RAG-Token models? | RAG-Sequence uses one document for the whole sequence; RAG-Token can use a different document per token |
| 10 | In the FEVER analysis by Lewis et al., how often is the top retrieved document from a gold article, and how often is a gold article in the top 10? | 71% and 90% |

### Measurement

`T_retriever` and `T_generator` were measured with `time.perf_counter()`. `T_in` is query tokens plus retrieved chunk tokens; the full prompt adds about 70 tokens of instructions and chat formatting.

CO2e came from CodeCarbon's offline tracker, set to the Alberta grid (628 g CO2e/kWh in these runs) and sampling every second. A single query is shorter than that sampling interval, and early tests recorded zero CPU energy as a result. Each of the 20 rows in `results/results.csv` is therefore the per-query mean of 10 consecutive executions inside one tracker. A warm-up query was discarded for each configuration. The generated answers were identical in the main and reversed-order runs.

Answers were graded Pass/Fail against the expected answer using a shuffled sheet that hid the configuration. A pass had to be correct and grounded in the source document.

Three additional measurements support the analysis:

- **Reversed-order run:** repeats both configurations in the opposite order to show run-to-run variation.
- **k sweep:** tests k = 1, 3, 5 and 8 for each embedding model and chunk size.
- **Retrieval energy test:** runs 2,000 searches per configuration, because one search is too short to measure.

## b) Results

### Average across the 10 questions

| Metric | Baseline | Optimized | Change |
|---|---|---|---|
| `T_retriever` (ms) | 4.94 | 8.09 | +64% |
| `T_generator` (ms) | 983 | 865 | −12.0% |
| `T_total` (ms) | 988 | 873 | −11.6% |
| `T_in` (tokens) | 2,062 | 1,541 | −25.3% |
| Output tokens | 34.9 | 31.8 | −8.9% |
| Energy per query (J) | 206.9 | 189.7 | −8.3% |
| CO2e per query (kg) | 3.61e-05 | 3.31e-05 | −8.3% |
| **Accuracy** | **9/10** | **6/10** | **−3 questions** |
| **Passes per second of `T_total`** | **0.911** | **0.687** | **−25%** |
| **Passes per gram CO2e** | **25** | **18** | **−27%** |
| **Energy per correct answer (J)** | **229.9** | **316.2** | **+38%** |

In the reversed-order run, Baseline used 221.8 J and 1,006 ms per query, and Optimized used 188.2 J and 872 ms. Across the two runs, Optimized therefore saved 8–15% of energy and 12–13% of total time. GPU energy was 95% of the total for both configurations, with RAM at about 3% and CPU at 2%.

### Results per question

| # | Baseline | Optimized | Baseline energy (J) | Optimized energy (J) | Facts retrieved by Optimized? |
|---|---|---|---|---|---|
| 1 | Pass | Pass | 143.7 | 143.9 | Yes |
| 2 | Pass | Fail ("I don't know") | 101.0 | 66.7 | No |
| 3 | Pass | Pass | 240.5 | 256.2 | Yes |
| 4 | Pass | Pass | 200.9 | 170.2 | Yes |
| 5 | Pass | Pass | 177.4 | 178.3 | Yes |
| 6 | Pass | Fail ("I don't know") | 103.6 | 69.3 | No |
| 7 | Fail (invented meaning of FPO) | Fail (invented meaning of FPO) | 309.8 | 377.8 | No |
| 8 | Pass | Pass | 208.4 | 172.7 | Yes |
| 9 | Pass | Pass | 351.9 | 394.6 | Yes |
| 10 | Pass | Fail ("I don't know") | 232.0 | 67.2 | No |

Baseline retrieved the required facts for every question except question 7. "Facts retrieved" comes from a keyword check in `check_facts.py`, and its pass/fail pattern matched the grades on all 20 runs.

### Effect of k alone

| Embedding / chunk size | k | `T_in` | Output tokens | `T_generator` (ms) | Energy (J) | Facts retrieved | Facts in answer |
|---|---|---|---|---|---|---|---|
| MiniLM / 256 | 1 | 279 | 19.7 | 483 | 101.2 | 3/10 | 4/10 |
| MiniLM / 256 | 3 | 791 | 28.9 | 726 | 157.1 | 6/10 | 6/10 |
| MiniLM / 256 | 5 | 1,303 | 35.2 | 919 | 202.2 | 6/10 | 6/10 |
| MiniLM / 256 | 8 | 2,062 | 34.9 | 995 | 218.6 | 9/10 | 9/10 |
| BGE / 512 | 1 | 535 | 22.6 | 560 | 121.0 | 4/10 | 3/10 |
| BGE / 512 | 3 | 1,541 | 31.8 | 871 | 188.5 | 6/10 | 6/10 |
| BGE / 512 | 5 | 2,558 | 32.7 | 967 | 217.8 | 8/10 | 8/10 |
| BGE / 512 | 8 | 4,092 | 36.8 | 1,215 | 275.7 | 10/10 | 9/10 |

The last two columns are keyword checks, not hand grades. A least-squares fit over all 80 sweep rows gave:

- `T_generator` = −19.5 ms + 88.4 ms per 1,000 input tokens + 23.43 ms per output token (R² = 0.992)
- Energy = −0.1 J + 24.5 J per 1,000 input tokens + 4.73 J per output token (R² = 0.998)

## c) Analysis and Conclusion

### The trade-off: k and generation time

Passing more context does increase the cost significantly, but less than proportionally. Raising k from 1 to 8 multiplied input tokens by about 7.5, while generation time and energy only rose by 2.1–2.3 times. Part of that rise comes from longer answers, since small k often produced a short "I don't know".

The fit separates the two effects. An extra 1,000 input tokens added about 88 ms and 24.5 J, while one output token cost 23.4 ms and 4.73 J. In energy terms, one generated token cost about the same as 190 input tokens.

Under the suggested settings, the context reduction is also smaller than the labels imply. Eight 256-token chunks and three 512-token chunks differ by only 25%, or about 520 tokens. The fit predicts this saves about 46 ms and 13 J per query, which is 6% of Baseline's energy.

### Energy-accuracy assessment

**Baseline achieved the best efficiency score.** It delivered 0.911 passes per second against 0.687, and 25 passes per gram of CO2e against 18. Baseline stays ahead even when using its highest measured energy: 246 J per correct answer against 314 J. At its measured energy, Optimized would have needed 8–9 passes out of 10 to match it.

**Optimized's headline saving is partly a side effect of failing.** Its three "I don't know" answers took about 67 J each, against an average of 146 J for Baseline's correct answers to the same questions. Those three questions account for more than three-quarters of the mean energy saving in every measurement. On the six questions both configurations passed, the saving was 0.5–9% of energy and 4–7% of generation time, which fits the small context reduction.

**The energy spent on the better embedding model was negligible.**

| Cost | Baseline | Optimized | Extra for Optimized |
|---|---|---|---|
| Retrieval, per query | 0.27 J | 0.47 J | 0.20 J |
| Building the index, one time | 19.0–26.0 J | 36.3–37.2 J | 10–18 J |

Retrieval used about 0.1% of Baseline's query energy and 0.25% of Optimized's. Even using the smallest shared-question saving, the extra indexing energy is recovered in fewer than 20 queries, so the extra compute was affordable.

**However, that compute did not deliver enough relevance at k = 3.** At k = 3, both embedding and chunk-size combinations retrieved the required facts for 6 of 10 questions, and MiniLM with 256-token chunks did so using less energy (157 J against 188.5 J). BGE with 512-token chunks did retrieve more at k = 5 (8/10 against 6/10) and k = 8 (10/10 against 9/10). Its k = 5 setting used about the same energy as Baseline's k = 8 (218 J against 219 J), yet got one fewer fact into its answers (8/10 against 9/10). The sweep changed the embedding model and chunk size together, so it can't show which of the two made the difference.

Every failed run was a retrieval miss. In three of the five, the model correctly declined to answer. In the two question 7 runs, the retrieved text lacked the phrase "floating point", and the model invented a meaning for FPO instead ("Financial Price of a Paper" and "Flops Per Output"). This shows that a retrieval miss can produce a confident wrong answer, not only a safe refusal.

### Final recommendation

**Based purely on this data, the recommended architecture is the Baseline configuration: `all-MiniLM-L6-v2`, 256-token chunks with 50 overlap, k = 8, and the same 7B generator.** It had the highest accuracy, the most passes per second and per gram of CO2e, and the lowest energy per correct answer. It costs 8–15% more energy per query, but that extra energy buys three more correct answers out of ten.

The main lesson is that retrieval energy is too small to drive this decision. What matters is whether the required text reaches the generator, and how much that context and the answer length add to generation.

### Limitations

- **Small sample:** ten questions over three papers means one question moves accuracy by 10 points, so these results apply to this task and don't establish a general ranking.
- **Question selection:** several questions need one specific number or definition. This favours high k and may not reflect other workloads.
- **Energy variation:** Baseline's energy ranged from 207 to 222 J per query across the three measurements, and Optimized's from 188 to 190 J. Small differences should not be over-read.
- **CodeCarbon estimates:** the figures use a fixed grid intensity, estimate CPU power, and exclude data-centre overhead (PUE).
- **Chunk truncation:** chunks were sized in generator tokens, but the embedding models count differently. 58 of 101 Baseline chunks and 20 of 45 Optimized chunks exceeded the embedder's limit, by at most 35 and 52 tokens. The 50-token overlap repeats nearly all of that cut text at the start of the next chunk.
- **Sweep grading:** the k sweep used keyword checks only, so it shows a pattern rather than graded accuracy.
