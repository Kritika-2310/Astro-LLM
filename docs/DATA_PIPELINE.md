# v2 data pipeline: audience-adaptive astronomy Q&A

Goal: ~300-400 questions, each answered in 3 audience styles (beginner / enthusiast / advanced), grounded in OpenStax Astronomy 2e + Wikipedia, generated with Gemini's free tier and filtered by a judge. v1 data (`data/processed/`, 159 pairs) is untouched.

## Run order (Codespaces, CPU only)

```bash
pip install -r requirements-dev.txt
# download OpenStax Astronomy 2e PDF -> data/raw/astronomy2e.pdf (not committed)

python scripts/data_pipeline/01_extract_corpus.py --pdf data/raw/astronomy2e.pdf --list-only   # check chapter list
python scripts/data_pipeline/01_extract_corpus.py --pdf data/raw/astronomy2e.pdf --wiki-tsv data/raw/wiki_topics.tsv
python scripts/data_pipeline/02_generate_qa.py --limit 200      # repeat until target reached (resume-safe)
python scripts/data_pipeline/03_filter.py --skip-judge          # sanity-check rule/dedupe drop counts
python scripts/data_pipeline/03_filter.py --limit 150           # repeat until all judged
python scripts/data_pipeline/04_split.py
python scripts/data_pipeline/05_make_mcq.py --limit 80          # held-out-chapter MCQs; then review by hand
```

Training stays on Colab: `python scripts/train.py --config configs/training_config_v2.yaml`.

## Design rules

- Held-out chapters (config `corpus.heldout_chapters`; random seeded fallback via `n_heldout_chapters`) never feed training data; their Wikipedia pages are held out too. If you change them, re-run step 1, and delete stale `data/interim/` and `data/eval/heldout_mcq.jsonl`.
- Split is grouped by source chunk, so all styles of a question (and any passage) stay in one split; `04_split.py` asserts this.
- Generated questions too similar to `data/eval_benchmark.jsonl` or v1 val/test are dropped (`leak_threshold`).
- A question is kept only if all 3 style answers pass the rule filters and the judge.

## Report in the paper

`data/interim/filter_report.json` (counts per stage, readability per style), `data/processed_v2/split_manifest.json`, `data/corpus/heldout_chapters.json`, seed, model names, `prompts/audience_styles.yaml`, judge prompt (in `03_filter.py`), and the result of a manual audit of ~50 random rows.
