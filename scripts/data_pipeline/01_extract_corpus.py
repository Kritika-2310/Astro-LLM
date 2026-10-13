"""
Step 1: Build the source corpus (OpenStax Astronomy 2e + Wikipedia) and hold out whole chapters.

Usage:
    python scripts/data_pipeline/01_extract_corpus.py --pdf data/raw/astronomy2e.pdf --list-only   # inspect chapters
    python scripts/data_pipeline/01_extract_corpus.py --pdf data/raw/astronomy2e.pdf \
        --wiki-tsv data/raw/wiki_topics.tsv

wiki_topics.tsv: one line per page -> <Wikipedia page title><TAB><OpenStax chapter number>
    The chapter number decides whether the page is held out (pages of held-out chapters never reach training).

Outputs (data/corpus/):
    chunks_train.jsonl, chunks_heldout.jsonl, heldout_chapters.json
"""
import argparse
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_cfg, read_jsonl, write_jsonl  # noqa: E402

import json  # noqa: E402


def clean(text):
    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s or "openstax.org" in s.lower():
            continue
        if re.fullmatch(r"\d{1,4}", s):          # bare page numbers
            continue
        lines.append(s)
    out = " ".join(lines)
    return re.sub(r"\s+", " ", out)


def chunk_words(text, size, overlap):
    words = text.split()
    step = max(size - overlap, 1)
    return [" ".join(words[i:i + size]) for i in range(0, len(words), step)]


def openstax_chapters(pdf_path, regex):
    import fitz
    doc = fitz.open(pdf_path)
    toc = [t for t in doc.get_toc() if t[0] == 1]
    if not toc:
        sys.exit("No level-1 bookmarks in the PDF. Use the web-PDF from openstax.org or adapt this function.")
    chapters = [(title.strip(), start) for _, title, start in toc if re.match(regex, title.strip())]
    result = []
    for i, (title, start) in enumerate(chapters):
        nxt = chapters[i + 1][1] if i + 1 < len(chapters) else None
        end = (nxt - 1) if nxt else len(doc)
        text = clean(" ".join(doc[p].get_text() for p in range(start - 1, end)))
        result.append((i + 1, title, text))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--wiki-tsv", default=None)
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument("--cfg", default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    c = cfg["corpus"]
    chapters = openstax_chapters(args.pdf, c["openstax_chapter_regex"])
    print(f"Found {len(chapters)} chapters:")
    for n, title, text in chapters:
        print(f"  {n:>2}  {title[:70]:<70} {len(text.split()):>7} words")
    if args.list_only:
        print("\nIf this list looks wrong (front matter, appendices), tune corpus.openstax_chapter_regex in configs/data_pipeline.yaml.")
        return

    all_ids = [n for n, _, _ in chapters]
    fixed = c.get("heldout_chapters") or []
    if fixed:
        missing = [n for n in fixed if n not in all_ids]
        if missing:
            sys.exit(f"heldout_chapters {missing} not found in this PDF (chapters are 1..{len(all_ids)}).")
        heldout_ids = sorted(fixed)
    else:
        rng = random.Random(cfg["seed"])
        heldout_ids = sorted(rng.sample(all_ids, c["n_heldout_chapters"]))
    title_of = {n: t for n, t, _ in chapters}

    records = []
    for n, title, text in chapters:
        for j, ch in enumerate(chunk_words(text, c["chunk_words"], c["chunk_overlap"])):
            if len(ch.split()) < c["min_chunk_words"]:
                continue
            records.append({"chunk_id": f"openstax-{n:02d}-{j:04d}", "source": "openstax_astronomy_2e",
                            "chapter_idx": n, "chapter": title, "text": ch})

    if args.wiki_tsv:
        import wikipediaapi
        wiki = wikipediaapi.Wikipedia(
            user_agent="AstroSagan-research/0.1 (https://github.com/Kritika-2310/Astro-LLM)", language="en")
        for ln in Path(args.wiki_tsv).read_text().splitlines():
            if not ln.strip() or ln.startswith("#"):
                continue
            page_title, ch_no = [x.strip() for x in ln.split("\t")]
            ch_no = int(ch_no)
            page = wiki.page(page_title)
            if not page.exists():
                print(f"  wiki: missing page {page_title!r}")
                continue
            text = re.sub(r"\s+", " ", page.text)
            for j, ch in enumerate(chunk_words(text, c["chunk_words"], c["chunk_overlap"])):
                if len(ch.split()) < c["min_chunk_words"]:
                    continue
                slug = re.sub(r"\W+", "-", page_title.lower()).strip("-")
                records.append({"chunk_id": f"wiki-{slug}-{j:03d}", "source": "wikipedia",
                                "chapter_idx": ch_no, "chapter": title_of.get(ch_no, f"chapter {ch_no}"),
                                "wiki_title": page_title, "text": ch})

    train = [r for r in records if r["chapter_idx"] not in heldout_ids]
    held = [r for r in records if r["chapter_idx"] in heldout_ids]
    out = ROOT / "data" / "corpus"
    write_jsonl(out / "chunks_train.jsonl", train)
    write_jsonl(out / "chunks_heldout.jsonl", held)
    (out / "heldout_chapters.json").write_text(json.dumps(
        {"seed": cfg["seed"], "heldout": [{"idx": n, "title": title_of[n]} for n in heldout_ids]}, indent=2))
    print(f"\nHeld-out chapters: {[(n, title_of[n]) for n in heldout_ids]}")
    print(f"chunks_train: {len(train)}   chunks_heldout: {len(held)}   -> {out}")

    stale = [r for r in read_jsonl(ROOT / "data/interim/raw_qa.jsonl") if r["chapter_idx"] in heldout_ids]
    if stale:
        print(f"\nWARNING: data/interim/raw_qa.jsonl has {len(stale)} rows from now-held-out chapters. "
              "03_filter.py will drop them automatically; to start clean: rm -rf data/interim")
    for stale_file in ("data/eval/heldout_mcq.jsonl", "data/interim/mcq_rejected.txt"):
        if (ROOT / stale_file).exists():
            print(f"NOTE: {stale_file} was built for the previous held-out set; delete it and re-run step 5.")


if __name__ == "__main__":
    main()
