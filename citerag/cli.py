"""Command line: citerag ingest | search | ask | eval | eval-answers | serve"""

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


def _progress(i: int, total: int, it: dict) -> None:
    q = it["question"] if len(it["question"]) <= 40 else it["question"][:39] + "…"
    name = it["generator"] or "?"
    print(f"[{i}/{total}] {name}  {it['latency_ms']:.0f} ms  {it['slice']}  {q}", file=sys.stderr, flush=True)


def _verbose_item(it: dict) -> None:
    judge = it.get("judge")
    claims = [c for s in judge["segments"] for c in s.get("claims", [])] if judge else []
    faith = f"{sum(c['supported'] for c in claims)}/{len(claims)}" if judge else "-"
    rel = (judge["relevance"].get("score", "error")) if judge else "-"
    print(f"    slice={it['slice']} cited={it['cited_files']} invalid_markers={it['invalid_markers']} "
          f"faithfulness={faith} relevance={rel}", file=sys.stderr)
    print(f"    {it['text'][:120]!r}", file=sys.stderr, flush=True)


def cmd_eval_answers(args) -> int:
    from . import answer_eval as ae
    from .engine import load_pipeline
    from .evaluation import evaluate_gate, load_golden
    from .generation import ExtractiveGenerator, OllamaGenerator

    settings = _settings(args)
    names = list(dict.fromkeys(args.generator or ["extractive"]))
    judge_model = args.judge_model or settings.ollama_model
    needed = ([settings.ollama_model] if "ollama" in names else []) + ([judge_model] if args.judge == "ollama" else [])
    if needed:
        problem = ae.check_ollama(settings.ollama_url, needed)
        if problem:
            print(f"error: {problem}", file=sys.stderr)
            return 1

    golden = load_golden(args.golden)
    if args.limit is not None:
        kept, n = [], 0
        for g in golden:
            if not g.relevant:
                kept.append(g)
            elif n < args.limit:
                kept.append(g)
                n += 1
        golden = kept
    generators = {
        "extractive": ExtractiveGenerator(),
        **({"ollama": OllamaGenerator(model=settings.ollama_model, url=settings.ollama_url)}
           if "ollama" in names else {}),
    }
    judge = ae.OllamaJudge(model=judge_model, url=settings.ollama_url) if args.judge == "ollama" else None

    pipeline = load_pipeline(settings, generator=generators[names[0]])
    gate = evaluate_gate(pipeline, golden)
    all_items: dict[str, list[dict]] = {}
    try:
        for name in names:
            pipeline.generator = generators[name]
            items = all_items[name] = []
            for i, g in enumerate(golden, 1):
                it = ae.run_item(pipeline, g)
                if judge:
                    ae.judge_item(judge, it)
                items.append(it)
                _progress(i, len(golden), it)
                if args.verbose:
                    _verbose_item(it)
    except ae.JudgeUnavailable as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    results = {name: ae.summarise(items, gate) for name, items in all_items.items()}
    print(ae.format_answer_table(results))
    answerable = gate["tp"] + gate["fn"]
    unanswerable = gate["tn"] + gate["fp"]
    print(f"abstain gate: answered {gate['tp']}/{answerable} answerable, "
          f"refused {gate['tn']}/{unanswerable} unanswerable")
    for name, r in results.items():
        label = f" [{name}]" if len(results) > 1 else ""
        o = r["overall"]
        print(f"model declined (heuristic): {o['model_declined']} answered questions{label}")
        if judge:
            f, rel = o.get("faithfulness"), o.get("relevance_mean")
            print(f"judge ({judge_model}, prompts v{ae.JUDGE_PROMPT_VERSION}){label}: "
                  f"faithfulness {'n/a' if f is None else format(f, '.3f')} ({o.get('claims', 0)} claims), "
                  f"relevance {'n/a' if rel is None else format(rel, '.2f')}, "
                  f"judge errors {o.get('judge_errors', 0)}/{o.get('judge_calls', 0)}")
    sanity = None
    if judge:
        try:
            sanity = ae.judge_sanity(judge)
        except ae.JudgeUnavailable as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        print(f"judge sanity: caught {sanity['caught']}/{sanity['bad']} bad answers, "
              f"{sanity['false_alarms']} false alarms on {sanity['good']} good ones")

    if not args.no_write:
        config = {
            "top_k": settings.top_k, "mode": settings.mode, "min_coverage": settings.min_coverage,
            "min_dense": settings.min_dense,
            "generators": {n: (settings.ollama_model if n == "ollama" else None) for n in names},
            "judge": ae.judge_info(judge_model) if judge else None,
        }
        if args.limit is not None:
            config["limit"] = args.limit
        payload = {
            "date": ae.date.today().isoformat(), "git_commit": ae.git_commit(), "config": config,
            "results": results, "judge_sanity": sanity,
            "items": all_items,
        }
        out = ae.write_results(args.out or ae.default_results_path(), payload)
        print(f"wrote {out}")
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

    s = sub.add_parser("eval-answers", help="answer quality: citations, latency and an optional local LLM judge")
    s.add_argument("--generator", action="append", choices=["extractive", "ollama"],
                   help="repeatable (default: extractive). Claude is deliberately not offered: no paid API calls")
    s.add_argument("--judge", choices=["none", "ollama"], default="none")
    s.add_argument("--judge-model", default=None, help="Ollama model for the judge (default: ollama_model)")
    s.add_argument("--golden", default="eval/golden.jsonl")
    s.add_argument("--limit", type=int, default=None, help="first N answerable questions plus all unanswerable ones")
    s.add_argument("--out", default=None, help="results JSON (default eval/results/answers-DATE.json)")
    s.add_argument("--no-write", action="store_true", help="don't write the results file")
    s.add_argument("-v", "--verbose", action="store_true")
    s.set_defaults(fn=cmd_eval_answers)

    s = sub.add_parser("serve", help="run the web app")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)

    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except ValueError as e:
        if not str(e).startswith("unknown generator"):
            raise
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
