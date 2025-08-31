import sys, json, argparse

"""Simple preview CLI for papers.jsonl — prints key fields for QA."""


def preview(path: str, limit: int = 10):
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            row = json.loads(line)
            print(f"[{i}] id={row.get('id') or row.get('entry_id')} title={row.get('title')}")
            # print a few derived fields if present
            for k in ('summary_short','novelty_score','symbolic_representation','core_claims'):
                if k in row:
                    print(f"   {k}: {str(row.get(k))[:200]}")
            print()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('papers_jsonl')
    ap.add_argument('--limit', type=int, default=10)
    args = ap.parse_args()
    preview(args.papers_jsonl, limit=args.limit)
