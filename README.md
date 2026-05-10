# Vector_Search_NLP

## Training data (MS MARCO passage ranking v1.1)

The project uses **`microsoft/ms_marco` · config `v1.1`** from Hugging Face: real web-search-style **queries** and **passages** with binary relevance (`is_selected`). Each JSONL line is one **(query, document)** pair where `document` is a relevant passage.

| Split        | Role        | Query–passage pairs |
|-------------|-------------|---------------------|
| `train`     | Training    | 88,523              |
| `validation` | Validation | 10,783              |
| `test`      | Held-out eval | 10,448            |

**Regenerate full files** (writes large `*.jsonl` under `data/msmarco_pairs/`, gitignored except samples):

```bash
pip install -r requirements.txt
python scripts/build_msmarco_pairs.py
```

Options: `--max_train N`, `--max_validation N`, `--max_test N` for smaller subsets. **Committed** in-repo: `data/msmarco_pairs/train.sample.jsonl` (80 lines) and `manifest.json`.

## TF-IDF baseline

Retrieval uses **sklearn `TfidfVectorizer`** (unigrams, `max_features=65536`, `sublinear_tf=True`) over the **union of unique passages** from `train.jsonl`, `validation.jsonl`, and `test.jsonl` (so each labeled passage exists in the index). Evaluation is on **distinct `query_id`s** in `test.jsonl`; gold labels are all passages paired with that query in the test split. Metrics use the **best rank** among multiple positives. Ties in similarity are broken by **lower passage index** in the fixed corpus list (deterministic).

```bash
python scripts/run_tfidf_baseline.py
```

From the project root this works without setting `PYTHONPATH` (the script adds the repo root to `sys.path`). Default output walks through raw samples, vectorizer stats, and one example query (tokens, sparse query weights, top‑5 hits vs gold). Use `--quiet` for metrics‑only JSON; `--limit_queries N` for a quick subset.

**Current run (full test queries, 9,345 queries):**

| Metric | Value |
|--------|------:|
| MRR | 0.518 |
| Recall@1 | 0.404 |
| Recall@5 | 0.657 |
| Recall@10 | 0.742 |
| Recall@50 | 0.877 |
| Recall@100 | 0.908 |

`--limit_queries N` runs a subset; `--ngram_max 2` enables bigrams (slower, larger vocabulary); `--max_features` controls the vectorizer cap.
