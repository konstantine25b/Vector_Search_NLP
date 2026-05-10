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
