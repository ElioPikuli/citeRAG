"""Command line: citerag ingest | search | ask | eval | serve"""

from __future__ import annotations

import argparse
import sys
import textwrap

from .config import Settings


def _settings(args) -> Settings:
    return Settings.load(args.config)


def cmd_ingest(args) -> int:
    from .engine import ingest

    _, r = ingest(_settings(args), rebuild=args.rebuild)
    print(
        f"indexed {r.files_total} files ({r.files_changed} changed, {r.files_removed} removed) -> "
        f"{r.chunks_total} chunks, {r.chunks_embedded} embedded in {r.seconds}s"
    )
    return 0


def cmd_search(args) -> int:
    from .engine import load_pipeline

    p = load_pipeline(_settings(args))
    for rank, h in enumerate(p.retriever.search(args.query, args.k, mode=args.mode), start=1):
        print(f"{rank:>2}. {h.chunk.label}  [{h.chunk.title}]")
        print(f"    score={h.score:.4f} bm25_rank={h.bm25_rank} dense_rank={h.dense_rank} cos={h.dense:.3f}")
    return 0


def cmd_ask(args) -> int:
    from .engine import load_pipeline

    p = load_pipeline(_settings(args))
    hits: list[dict] = []
    for ev in p.stream(args.question):
        if ev["type"] == "sources":
            hits = ev["hits"]
        elif ev["type"] == "token":
            print(ev["text"], end="", flush=True)
        elif ev["type"] == "cite":
            print("".join(f"[{i + 1}]" for i in ev["hits"]), end="", flush=True)
        elif ev["type"] == "error":
            print(f"\nerror: {ev['message']}", file=sys.stderr)
            return 1
        elif ev["type"] == "done":
            print(f"\n\n({ev['generator']})")
    if hits:
        print("\nSources:")
        for i, h in enumerate(hits, start=1):
            print(f"  [{i}] {h['label']}  {textwrap.shorten(h['title'], 60)}")
    return 0


def cmd_eval(args) -> int:
    from .engine import load_pipeline
    from .evaluation import evaluate, evaluate_gate, format_table, load_golden

    p = load_pipeline(_settings(args))
    golden = load_golden(args.golden)
    results = {mode: evaluate(p.retriever, golden, mode=mode, k=args.k) for mode in ("bm25", "dense", "hybrid")}
    print(format_table(results, args.k))
    gate = evaluate_gate(p, golden)
    print(
        f"abstain gate: answered {gate['answered_when_answerable']:.0%} of answerable questions, "
        f"abstained on {gate['abstained_when_unanswerable']:.0%} of unanswerable ones"
    )
    if args.verbose:
        for q in results["hybrid"].per_query:
            mark = "ok " if q["hit"] else "MISS"
            print(f"{mark} rr={q['rr']:.2f}  {q['question']}")
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    from .api import app_from_settings

    uvicorn.run(app_from_settings(args.config), host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="citerag", description="Hybrid-search RAG with grounded citations")
    ap.add_argument("--config", default="citerag.toml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest", help="index the corpus (incremental)")
    s.add_argument("--rebuild", action="store_true", help="ignore the existing index")
    s.set_defaults(fn=cmd_ingest)

    s = sub.add_parser("search", help="retrieval only")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=6)
    s.add_argument("--mode", choices=["hybrid", "bm25", "dense"], default=None)
    s.set_defaults(fn=cmd_search)

    s = sub.add_parser("ask", help="answer a question with citations")
    s.add_argument("question")
    s.set_defaults(fn=cmd_ask)

    s = sub.add_parser("eval", help="retrieval metrics on the golden set")
    s.add_argument("--golden", default="eval/golden.jsonl")
    s.add_argument("-k", type=int, default=5)
    s.add_argument("-v", "--verbose", action="store_true")
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("serve", help="run the web app")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
