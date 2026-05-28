# Vector_Search_NLP

## Local setup

This repo expects Python 3. Use a virtual environment so dependency versions stay isolated:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

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

`--limit_queries N` runs a subset; `--ngram_max 2` enables bigrams (slower, larger vocabulary); `--max_features` controls the vectorizer cap.

## Bag-of-words cosine baseline

Second lexical baseline: **`sklearn.CountVectorizer`** (word counts, same `max_features` / `ngram` defaults as TF‑IDF), then **row-wise L2 normalization** so the **dot product** between a query row and each document row equals **cosine similarity** between raw BoW count vectors. Same corpus union and **same** `evaluate_ranking` protocol as TF‑IDF. Optional **`--binary`**: presence/absence BoW instead of integer counts.

```bash
python scripts/run_bow_baseline.py
```

Use `--quiet`, `--limit_queries N`, and the same tokenizer-style flags as TF‑IDF where applicable (`--max_features`, `--ngram_min`, `--ngram_max`).

## Retrieval metrics comparison (test set)

All methods below use the **same** setup: deduplicated passage index from train+val+test (**105,793** unique passages), evaluation on **9,345** distinct test `query_id`s, **MRR** and **Recall@k** with **best rank** among gold passages.

| Metric | TF-IDF | BoW cosine | Dense bi-encoder |
|--------|-------:|-----------:|-----------------:|
| MRR | 0.518 | 0.197 | 0.356 |
| Recall@1 | 0.404 | 0.135 | 0.256 |
| Recall@5 | 0.657 | 0.257 | 0.467 |
| Recall@10 | 0.742 | 0.320 | 0.554 |
| Recall@50 | 0.877 | 0.464 | 0.731 |
| Recall@100 | 0.908 | 0.528 | 0.792 |

- **TF-IDF**: `python scripts/run_tfidf_baseline.py` — unigrams, `max_features=65536`, `sublinear_tf=True`.
- **BoW cosine**: `python scripts/run_bow_baseline.py` — `CountVectorizer` + row L2 normalize (integer counts, no IDF).
- **Dense bi-encoder**: `python scripts/eval_dense_retriever.py --checkpoint checkpoints/dense_msmarco/last.pt` — DistilBERT, **1 epoch**, batch 32, InfoNCE training.

TF-IDF is the strongest lexical baseline here (IDF down-weights common terms). Raw BoW cosine is weaker but still a valid second baseline. The dense model is below TF-IDF after limited training; longer training and larger batches (more in-batch negatives) are the main levers to improve it.

## Book chunk search demo

The final project demo needs to retrieve 200-300 word passages from **Dan Jurafsky and James H. Martin — Speech and Language Processing**. The PDF is converted to overlapping chunks with page metadata:

```bash
.venv/bin/python scripts/build_book_chunks.py
```

Default chunking uses 240-word windows with 200-word stride and skips the front matter/table of contents. Output is written to `data/book_chunks/slp_chunks.jsonl`.

Search those chunks with the TF-IDF baseline:

```bash
.venv/bin/python scripts/search_book_demo.py "what is word tokenization" --top_k 5
```

After dense training creates `checkpoints/dense_msmarco/last.pt`, the same demo can search the book with the trained bi-encoder:

```bash
.venv/bin/python scripts/search_book_demo.py "what is word tokenization" --method dense --top_k 5
```

## InfoNCE (contrastive loss for dense retrieval)

Training uses **InfoNCE** over a mini-batch of \(B\) **(query, relevant passage)** pairs. Let \(q_i, d_i \in \mathbb{R}^H\) be **L2-normalized** embeddings for the \(i\)-th pair (so \(q_i^\top d_j\) is cosine similarity). With temperature \(\tau > 0\), logits are \(S_{ij} = (q_i^\top d_j) / \tau\). The **query→document** term is \(\operatorname{CrossEntropy}(S,\,\texttt{labels})\) with \(\texttt{labels}[i]=i\) (each row’s positive is the diagonal; off-diagonals are in-batch negatives). Optionally **symmetric** training averages that with **document→query**: \(\operatorname{CrossEntropy}(S^{\top},\,\texttt{labels})\). Implemented in `src/infonce_loss.py` (`infonce_loss`). This does **not** use the `sentence-transformers` library.

## Dense bi-encoder (shared Transformer)

Queries and passages are encoded by **`src/biencoder.BiEncoder`**: Hugging Face `AutoModel`, **mean** or **[CLS]** pooling, optional linear projection head, **L2-normalized** embeddings. Training minimizes **InfoNCE** on batches of positives from `train.jsonl`; symmetric query↔document InfoNCE is **on** by default (`--symmetric-infonce`). Checkpoints (`checkpoints/` is gitignored) store weights plus tokenizer/backbone ids.

```bash
# First run downloads DistilBERT weights from Hugging Face (~250MB).

# Quick sanity run (tiny subset — not a meaningful model):
python scripts/train_dense_retriever.py --max_train_samples 512 --epochs 1 --batch_size 16

# Full-ish training (still one epoch — tune epochs / LR / batch for your GPUs):
python scripts/train_dense_retriever.py --epochs 1 --batch_size 32

# Retrieval metrics aligned with TF‑IDF / BoW: same corpus union + grouped test queries
python scripts/eval_dense_retriever.py --checkpoint checkpoints/dense_msmarco/last.pt
```

See **Retrieval metrics comparison** above for TF-IDF vs BoW vs dense numbers.


## Margin contrastive loss (classical)

**Contrastive loss** (margin-based, batch of paired query/document embeddings): let \(D_{ij} = \|q_i - d_j\|_2\) (Euclidean distance in embedding space, typically after shared encoding without forcing normalization inside the loss). **Positive** term pulls matched pairs together: \(\frac{1}{B}\sum_i D_{ii}^2\). **Negative** term pushes non-matched in-batch pairs apart: average over all \(i \neq j\) of \(\max(0,\, m - D_{ij})^2\) with margin \(m \ge 0\). Implemented in `src/contrastive_loss.py` (`contrastive_loss`). It also does **not** use `sentence-transformers`.
