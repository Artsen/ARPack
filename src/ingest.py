from __future__ import annotations
import os, json, argparse
from typing import List, Dict
import arxiv
from dotenv import load_dotenv
from tqdm import tqdm
from src.schema import PaperRecord
from src.embed import get_embedding

load_dotenv()

def fetch_arxiv(category: str, limit: int = 50) -> List[Dict]:
    search = arxiv.Search(
        query=f"cat:{category}",
        max_results=limit,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    results = []
    for r in search.results():
        results.append({
            "entry_id": r.entry_id,
            "title": r.title.strip(),
            "authors": [a.name for a in r.authors],
            "categories": list(r.categories) if hasattr(r, "categories") else [],
            "summary": r.summary.strip(),
            "published": r.published.strftime("%Y-%m-%dT%H:%M:%SZ") if r.published else None,
            "updated": r.updated.strftime("%Y-%m-%dT%H:%M:%SZ") if r.updated else None,
            "doi": getattr(r, "doi", None),
            "journal_ref": getattr(r, "journal_ref", None),
            "links": {
                "pdf": r.pdf_url,
                "entry": r.entry_id,
            },
        })
    return results

def llm_summarize(record: Dict) -> Dict:
    """Summarize into rich fields. Uses OpenAI if available, else returns heuristics."""
    from os import getenv
    api_key = getenv("OPENAI_API_KEY")
    if not api_key:
        # Heuristic fallback
        summary = record["summary"]
        title = record["title"]
        return {
            "summary_short": (title[:180] + "...") if len(title) > 180 else title,
            "summary_bullets": [summary[:200] + ("..." if len(summary) > 200 else "")],
            "problem": None,
            "method": None,
            "results": None,
            "limitations": None,
            "key_terms": None,
            "key_quotes": None,
            "novelty_score": None,
            "open_questions": None,
            "hypothesis_seeds": None,
            "datasets": None,
            "code_repos": None,
            "eval_metrics": None,
        }
    # Use OpenAI for a structured JSON response
    from openai import OpenAI
    client = OpenAI()
    prompt = f"""You are a research assistant. Read the paper metadata below and produce a concise, structured JSON object with the following fields:
- summary_short (<=60 words)
- summary_bullets (3-6 bullets)
- problem (1-2 sentences)
- method (1-2 sentences)
- results (1-2 sentences; include metrics if present)
- limitations (optional)
- key_terms (<=8)
- key_quotes (<=3 direct quotes; <=25 words each, if any)
- novelty_score (0.0-1.0 subjective)
- open_questions (<=5)
- hypothesis_seeds (<=5; terse testable ideas)
- datasets (<=5 names or URLs if present)
- code_repos (<=5 URLs if present)
- eval_metrics (<=8 names)

Return ONLY valid JSON, no markdown.\n\nMETADATA:\n{json.dumps(record, ensure_ascii=False)}"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {"role": "system", "content": "You output only strict JSON. No commentary."},
            {"role": "user", "content": prompt},
        ],
    )
    import orjson
    content = resp.choices[0].message.content
    try:
        data = orjson.loads(content)
    except Exception:
        # Last resort: naive JSON recovery
        content = content.strip().strip("`").strip()
        data = json.loads(content)
    return data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat", required=True, help="arXiv category, e.g., cs.CL, cs.LG, cs.IR")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--out", default="data/papers.jsonl")
    args = ap.parse_args()

    rows = fetch_arxiv(args.cat, args.limit)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        for r in tqdm(rows, desc="Summarizing + embedding"):
            derived = llm_summarize(r)
            rec = PaperRecord(
                id=r["entry_id"].split("/")[-1],
                entry_id=r["entry_id"],
                title=r["title"],
                authors=r["authors"],
                categories=r["categories"],
                summary_raw=r["summary"],
                published=r["published"] or "",
                updated=r.get("updated"),
                doi=r.get("doi"),
                journal_ref=r.get("journal_ref"),
                links=r.get("links", {}),
                **derived,
                source_raw=r,
            )

            # Embed title + abstract (raw) + summary_short (if exists)
            texts = [rec.title, rec.summary_raw]
            if rec.summary_short:
                texts.append(rec.summary_short)
            import numpy as np
            vecs = get_embedding(texts)  # (N, D)
            fields = ["title", "abstract", "summary_short"]
            for i, name in enumerate(fields[:vecs.shape[0]]):
                rec.embeddings[name] = vecs[i].tolist()

            f.write(json.dumps(rec.model_dump(), ensure_ascii=False) + "\n")
    print(f"Wrote {args.out} with {len(rows)} records.")

if __name__ == "__main__":
    main()
