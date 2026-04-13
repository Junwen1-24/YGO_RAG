# YGO RAG

Local vector search plus OpenAI to answer Yu-Gi-Oh! rules questions from rule text and card data (experimental MVP).

## Requirements

- Python 3.10+
- OpenAI API key (**environment variables only**—do not hard-code or commit keys)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## API key

Export before you start the terminal session (example):

```bash
export OPENAI_API_KEY='sk-...'
```

Scripts in this repo **do not** read `.env`. If you keep keys in a file, load them yourself (e.g. `set -a; source .env; set +a` in your shell config) or use a tool like `direnv`.

## Build indexes

You need rule files under `ygo_rules/` and `data/card.json` (or pass your own paths). The Q&A index uses `data/ygo_eval_dataset.csv` by default (see `--qa-csv` / `--no-qa`).

```bash
export OPENAI_API_KEY='sk-...'
python preprocess_to_faiss.py
```

Optional flags: `--rules-dir`, `--cards-json`, `--rules-index-dir`, `--cards-index-dir`, `--qa-csv`, `--chunk-size`, `--embedding-model`.

Batch evaluation writes CSVs under `eval_results/` (see `run_batch_eval.py` / `run_judge_on_csv.py` defaults). Logs from past runs are kept under `logs/`.

`ygo_rules/**/*.json` files (e.g. glossary JSON under `build_site`) are parsed and included in the rules index.

## Usage

CLI:

```bash
export OPENAI_API_KEY='sk-...'
python query_ygo_judge.py --question "Your question"
```

Web (Streamlit):

```bash
export OPENAI_API_KEY='sk-...'
streamlit run app.py
```

**Streamlit Community Cloud:** the repo must include the built directories `faiss_rules_index/`, `faiss_cards_index/`, and (recommended) `faiss_qa_index/` from `preprocess_to_faiss.py`—they are not generated on the server. Set `OPENAI_API_KEY` in the app’s **Secrets**. Visitors do not paste a key.

The UI shows a disclaimer and the rule/card chunks retrieved for each answer; official events still follow KONAMI materials and the head judge.

## Disclaimer

This is not an official judge system. Answers are based on retrieved snippets and a large language model and may be wrong. For sanctioned play, follow the KONAMI rulebook, database, and head judge rulings.
