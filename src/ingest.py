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
    """Fetch up to `limit` papers from arXiv for `category` using the
    public HTTP API. This function ensures `start=0` and doesn't include
    an `id_list` parameter. It prefers `feedparser` for parsing and
    falls back to the `arxiv` library if needed.
    """
    import time
    import requests

    try:
        import feedparser
    except Exception:
        feedparser = None

    results: List[Dict] = []
    # Mirror the user's requested limit exactly (ensure at least 1)
    max_results = max(1, int(limit))

    base = "https://export.arxiv.org/api/query"
    params = {
        "search_query": f"cat:{category}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    attempts = 0
    text = None
    while attempts < 4:
        try:
            resp = requests.get(base, params=params, timeout=30)
            resp.raise_for_status()
            text = resp.text
            break
        except Exception as e:
            attempts += 1
            if attempts >= 4:
                print(f"[fetch_arxiv] search failed after {attempts} attempts: {e}")
                return []
            backoff = 0.8 * (2 ** (attempts - 1))
            print(f"[fetch_arxiv] request failed, retrying in {backoff:.1f}s (attempt {attempts}/4)...")
            time.sleep(backoff)

    entries = []
    if feedparser:
        feed = feedparser.parse(text)
        entries = feed.entries or []
    else:
        # fallback to arxiv.Client if feedparser isn't available
        try:
            from arxiv import Client

            client = Client()
            search = arxiv.Search(
                query=f"cat:{category}",
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending,
            )
            entries = list(client.results(search))
        except Exception as e:
            print(f"[fetch_arxiv] fallback parser failed: {e}")
            return []

    for e in entries:
        # Support both feedparser dict-like entries and arxiv.Entry objects
        if feedparser and hasattr(e, "get"):
            entry_id = e.get("id") or e.get("link")
            title = (e.get("title") or "").strip()
            # feedparser authors are list of dicts with 'name'
            authors = [a.get("name") for a in e.get("authors", []) if a.get("name")] if isinstance(e.get("authors", []), list) else []
            summary = (e.get("summary") or "").strip()
            published = e.get("published")
            updated = e.get("updated")
            # links may be list of dicts
            links = {"entry": entry_id}
            for L in e.get("links", []) or []:
                href = L.get("href")
                typ = L.get("type")
                if typ == "application/pdf" or (href and href.endswith(".pdf")):
                    links["pdf"] = href
                    break
            categories = [t.get("term") for t in (e.get("tags") or []) if t.get("term")] if e.get("tags") else []
        else:
            entry_id = getattr(e, "entry_id", None) or getattr(e, "id", None)
            title = (getattr(e, "title", "") or "").strip()
            authors = [a.name for a in getattr(e, "authors", [])]
            summary = getattr(e, "summary", "") or ""
            published = getattr(e, "published", None)
            updated = getattr(e, "updated", None)
            links = {"pdf": getattr(e, "pdf_url", None), "entry": entry_id}
            categories = list(getattr(e, "categories", [])) if hasattr(e, "categories") else []

        results.append({
            "entry_id": entry_id,
            "title": title,
            "authors": authors,
            "categories": categories,
            "summary": summary,
            "published": published,
            "updated": updated,
            "doi": None,
            "journal_ref": None,
            "links": links,
        })

        if len(results) >= limit:
            break

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
    # Use OpenAI for a structured JSON response. Be conservative and deterministic.
    from openai import OpenAI
    import time as _time
    import json as _orjson
    client = OpenAI()
    prompt = f"""You are a concise, expert research assistant. Read the paper metadata below and populate the provided structured schema (the system will provide a function schema); return values that match the schema exactly and use null for missing fields.

Please follow these rules when filling fields:
- Keep text fields brief and factual (summary_short <= 60 words).
- Use short lists where requested (e.g., summary_bullets 3-6 items, key_terms <=8).
- For equations or formal statements use LaTeX where appropriate (math_representation, symbolic_representation).
- For claim_confidences provide numbers in [0.0, 1.0] aligned positionally with core_claims.
- For counterfactuals and suggested_experiments give concise one-line items.
- For reproducibility_notes include whether code/data/seed information appears to be available.
- For computational_cost give a short qualitative estimate (e.g., "O(n log n)", "GPU hours: ~2").

Fields and what to put in them (short):
- summary_short: 1 concise paragraph (<=60 words)
- summary_bullets: 3-6 concise bullets
- problem: 1-2 sentences describing the scientific problem
- method: 1-2 sentences describing the approach
- results: 1-2 sentences summarizing key outcomes/metrics
- limitations: brief known limitations or failure modes
- key_terms: up to 8 important keywords
- key_quotes: up to 3 short direct quotes (<=25 words)
- novelty_score: subjective number 0.0-1.0
- open_questions: up to 5 open research questions
- hypothesis_seeds: up to 5 concise testable follow-ups
- datasets: dataset names or URLs referenced
- code_repos: URLs to code if present
- eval_metrics: metric names used in evaluation
- symbolic_representation: short symbolic / diagrammatic / LaTeX representation of the core idea
- core_claims: list of the explicit, testable claims made by the paper
- claim_confidences: list of floats [0-1] aligned with core_claims indicating confidence
- counterfactuals: brief counterfactuals / edge-cases where approach may break
- metacognitive_analysis: brief model reasoning about strengths/weaknesses of the paper's reasoning
- reproducibility_notes: notes about whether results look reproducible from provided materials
- suggested_experiments: short follow-up experiments
- math_representation: key equations or compact math (LaTeX)
- computational_cost: qualitative cost estimate (Big-O or runtime hints)
- data_bias_notes: potential dataset biases or distributional concerns
- implementation_tips: short practical tips for implementing the method

Return ONLY valid JSON (no markdown). Use short lists/strings where requested and prefer null for unknown fields.\n\nMETADATA:\n{json.dumps(record, ensure_ascii=False)}"""
    # Try function-calling (Responses API) with a strict JSON schema to force
    # the model to return structured output. If that fails, fall back to the
    # previous chat-based parsing approach.
    tools = [
        {
            "type": "function",
            "name": "paper_summary",
            "description": "Produce a structured JSON summary of the paper metadata. Return null for missing fields.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "summary_short": {"type": ["string", "null"], "description": "Concise summary (<=60 words)."},
                    "summary_bullets": {"type": ["array", "null"], "items": {"type": "string"}, "description": "3-6 bullet points."},
                    "problem": {"type": ["string", "null"], "description": "The main problem addressed."},
                    "method": {"type": ["string", "null"], "description": "High-level method description."},
                    "results": {"type": ["string", "null"], "description": "Key results or findings."},
                    "limitations": {"type": ["string", "null"], "description": "Known limitations."},
                    "key_terms": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Important keywords (<=8)."},
                    "key_quotes": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Direct quotes (<=3)."},
                    "novelty_score": {"type": ["number", "null"], "description": "Subjective novelty 0.0-1.0."},
                    "open_questions": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Open research questions."},
                    "hypothesis_seeds": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Short testable hypothesis seeds."},
                    "datasets": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Dataset names or URLs."},
                    "code_repos": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Repository URLs if any."},
                    "eval_metrics": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Evaluation metric names."},
                    # Extended meta fields for richer embeddings
                    "symbolic_representation": {"type": ["string", "null"], "description": "A short symbolic or formal representation of the core idea (math/graph/logic). Prefer LaTeX where applicable."},
                    "core_claims": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Explicit core claims or propositions (succinct)."},
                    "claim_confidences": {"type": ["array", "null"], "items": {"type": "number"}, "description": "Model's per-claim confidence 0.0-1.0 aligned with core_claims."},
                    "counterfactuals": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Brief counterfactual scenarios or edge cases where the method may fail."},
                    "metacognitive_analysis": {"type": ["string", "null"], "description": "Model's reasoning about where the paper's reasoning is weakest/strongest."},
                    "reproducibility_notes": {"type": ["string", "null"], "description": "Notes on reproducibility: data/code availability, seed/systems required."},
                    "suggested_experiments": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Follow-up experiments that would test the main claims."},
                    "math_representation": {"type": ["string", "null"], "description": "A concise mathematical formulation or key equations (LaTeX)."},
                    "computational_cost": {"type": ["string", "null"], "description": "Qualitative estimate of computational cost or complexity (e.g., Big-O, runtime)."},
                    "data_bias_notes": {"type": ["string", "null"], "description": "Potential dataset biases or population mismatches."},
                    "implementation_tips": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Practical tips for implementing or adapting the method."},
                },
                "required": [
                    "summary_short", "summary_bullets", "problem", "method", "results",
                    "limitations", "key_terms", "key_quotes", "novelty_score",
                    "open_questions", "hypothesis_seeds", "datasets", "code_repos", "eval_metrics",
                    "symbolic_representation", "core_claims", "claim_confidences", "counterfactuals",
                    "metacognitive_analysis", "reproducibility_notes", "suggested_experiments", "math_representation",
                    "computational_cost", "data_bias_notes", "implementation_tips"
                ],
                "additionalProperties": False,
            },
        }
    ]

    # build input messages
    input_list = [
        {"role": "system", "content": "You are a concise research assistant. Use the paper metadata to populate the paper_summary function schema exactly."},
        {"role": "user", "content": prompt},
    ]

    # attempt function-calling once
    try:
        resp = client.responses.create(
            model="gpt-4o-mini",
            input=input_list,
            tools=tools,
            tool_choice={"type": "function", "name": "paper_summary"},
            max_output_tokens=800,
            temperature=0.0,
        )

        # find a function call item in the response output
        func_args = None
        for item in getattr(resp, "output", []) or []:
            try:
                # item may be a dict-like or object with attributes
                it = item if isinstance(item, dict) else (item.model_dump() if hasattr(item, "model_dump") else None)
            except Exception:
                it = None
            if isinstance(it, dict):
                if it.get("type") in ("function_call", "tool_call"):
                    func_args = it.get("arguments")
                    break
            else:
                # try attribute-style access
                typ = getattr(item, "type", None)
                if typ in ("function_call", "tool_call"):
                    func_args = getattr(item, "arguments", None) or getattr(item, "arguments_json", None)
                    break

        if func_args:
            # arguments may be a JSON string
            import json as _json
            if isinstance(func_args, str):
                parsed = _json.loads(func_args)
            elif isinstance(func_args, dict):
                parsed = func_args
            else:
                parsed = _json.loads(str(func_args))

            # normalize: ensure all expected keys exist and have sensible types
            EXPECTED_KEYS = [
                "summary_short", "summary_bullets", "problem", "method", "results",
                "limitations", "key_terms", "key_quotes", "novelty_score",
                "open_questions", "hypothesis_seeds", "datasets", "code_repos", "eval_metrics",
                "symbolic_representation", "core_claims", "claim_confidences", "counterfactuals",
                "metacognitive_analysis", "reproducibility_notes", "suggested_experiments", "math_representation",
                "computational_cost", "data_bias_notes", "implementation_tips",
            ]

            def _ensure_list(x):
                if x is None:
                    return None
                if isinstance(x, list):
                    return x
                return [x]

            out = {}
            for k in EXPECTED_KEYS:
                v = parsed.get(k) if isinstance(parsed, dict) else None
                if v is None:
                    out[k] = None
                    continue
                if k in ("summary_bullets", "key_terms", "key_quotes", "open_questions", "hypothesis_seeds", "datasets", "code_repos", "eval_metrics", "core_claims", "counterfactuals", "suggested_experiments", "implementation_tips"):
                    out[k] = _ensure_list(v)
                elif k == "claim_confidences":
                    # list of numbers
                    arr = _ensure_list(v)
                    try:
                        out[k] = [float(x) for x in arr] if arr is not None else None
                    except Exception:
                        out[k] = None
                elif k == "novelty_score":
                    try:
                        out[k] = float(v)
                    except Exception:
                        out[k] = None
                else:
                    out[k] = v

            # align claim_confidences length to core_claims if both present
            if out.get("core_claims") and out.get("claim_confidences"):
                cc = out["core_claims"] or []
                conf = out["claim_confidences"] or []
                if len(conf) < len(cc):
                    # pad with None
                    conf = conf + [None] * (len(cc) - len(conf))
                elif len(conf) > len(cc):
                    conf = conf[: len(cc)]
                out["claim_confidences"] = conf

            return out

            # optional: validate against jsonschema if available
            try:
                import jsonschema
                schema = {
                    "type": "object",
                    "properties": {
                        k: {"type": ["array", "string", "number", "null"]} for k in EXPECTED_KEYS
                    },
                    "additionalProperties": True,
                }
                jsonschema.validate(instance=out, schema=schema)
            except Exception:
                # if validation fails or jsonschema missing, accept normalized output
                pass

            return out

    except Exception as e:
        # if any error, fall back to the older chat completion parsing approach below
        print(f"[llm_summarize] function-calling attempt failed: {e}")

    # Fallback: previous chat-based approach (attempt to recover a JSON blob)
    attempts = 2
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.0,
                max_tokens=800,
                messages=[
                    {"role": "system", "content": "You must output valid JSON only. Do not include any explanation or markdown."},
                    {"role": "user", "content": prompt},
                ],
            )
            content = getattr(resp.choices[0].message, "content", "")
            if not content:
                content = str(getattr(resp.choices[0], "text", ""))

            import re as _re
            import json as _json
            m = _re.search(r"(\{[\s\S]*\})", content)
            if not m:
                raise ValueError("no JSON found in model output")
            data = _json.loads(m.group(1))

            # normalize minimal
            return data
        except Exception as e:
            last_exc = e
            _time.sleep(0.5 * attempt)
            continue

    print(f"[llm_summarize] LLM fallback summarization failed: {last_exc}")
    summary = record.get("summary", "") or record.get("summary_raw", "")
    title = record.get("title", "")
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
