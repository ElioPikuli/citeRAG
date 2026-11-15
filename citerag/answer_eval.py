"""Answer-quality evaluation: run golden questions through retrieve + generate and score the answers.

Deterministic metrics (citations, markers, latency) are free. The optional LLM judge (a local Ollama model) checks
faithfulness per cited segment and answer relevance; its prompts live in judge_prompts.py. Nothing here calls a
paid API: the Claude generator is deliberately not reachable from this module's CLI.
"""

from __future__ import annotations

import http.client
import json
import math
import re
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import date
from pathlib import Path

from .evaluation import GoldenItem
from .generation import EXTRACTIVE_HEADER
from .judge_prompts import (
    FAITHFULNESS_SCHEMA,
    FAITHFULNESS_SYSTEM,
    FAITHFULNESS_USER,
    JUDGE_PROMPT_VERSION,
    RELEVANCE_SCHEMA,
    RELEVANCE_SCORES,
    RELEVANCE_SYSTEM,
    RELEVANCE_USER,
    RETRY_MESSAGE,
)

# Heuristic, not ground truth: phrases a model uses when it says the sources don't cover the question. It misses
# paraphrases and can match an answer that quotes such a phrase from a source, so treat counts as indicative.
DECLINED = re.compile(
    r"don['’]t cover|do not cover|doesn['’]t cover|does not cover|not covered|aren['’]t covered|no information|"
    r"couldn['’]t find|could not find|not mentioned|not (?:provided|included|detailed) in",
    re.IGNORECASE,
)


# ---- running one question ---------------------------------------------------------------------------

def _has_content(text: str) -> bool:
    return any(c.isalnum() for c in text)


def _merge(first: list[int], more: list[int]) -> list[int]:
    """`first` then the new entries of `more`: dedup, order kept."""
    return list(dict.fromkeys([*first, *more]))


def run_item(pipeline, item: GoldenItem) -> dict:
    """Run one question through retrieve + generate and record everything the metrics need.

    Consumes the events itself (not `pipeline.answer`) because the judge needs to know which text each citation
    covers. A `cite` event closes a segment: the text streamed since the previous cite, with those hits. A cite
    whose preceding text has no letter or digit (e.g. the model wrote "[6] text [3]" or "text [1]. [2]") is an
    orphan: its hits are carried into the next segment with content, or into the previous segment if none
    follows, so they are judged rather than silently dropped (counted in `orphan_cites`). If there is no segment
    at all they stay only in `cited`. Trailing text after the last cite becomes a segment with `hits: []` (or
    the pending orphan hits). `source_texts` maps each cited hit index to its chunk text, for the judge.
    """
    t0 = time.perf_counter()
    hits, gate = pipeline.retrieve(item.question)
    tokens: list[str] = []
    segments: list[dict] = []
    cited: list[int] = []
    pending = ""
    orphans: list[int] = []
    orphan_cites = 0
    done: dict = {}
    error: str | None = None
    for ev in pipeline.generate(item.question, hits, gate):
        kind = ev["type"]
        if kind == "token":
            tokens.append(ev["text"])
            pending += ev["text"]
        elif kind == "cite":
            valid = [i for i in ev["hits"] if 0 <= i < len(hits)]
            # The extractive header is a label, not a claim; judging it as part of the first quote counted it
            # as unsupported (measured in the first full run).
            pending = pending.replace(EXTRACTIVE_HEADER, "")
            if _has_content(pending) and valid:
                segments.append({"text": pending.strip(), "hits": _merge(orphans, valid)})
                orphans = []
            elif valid:
                orphans = _merge(orphans, valid)
                orphan_cites += 1
            pending = ""
            cited.extend(i for i in valid if i not in cited)
        elif kind == "done":
            done = ev
        elif kind == "error":
            error = ev["message"]
    if _has_content(pending):
        segments.append({"text": pending.strip(), "hits": orphans})
    elif orphans and segments:
        segments[-1]["hits"] = _merge(segments[-1]["hits"], orphans)
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    text = "".join(tokens)
    abstained = bool(done.get("abstained"))
    declined = (not abstained and error is None and bool(DECLINED.search(text)))
    return {
        "question": item.question,
        "slice": item.slice,
        "relevant": list(item.relevant),
        "answerable": bool(item.relevant),
        "abstained": abstained,
        "error": error,
        "generator": done.get("generator") or getattr(pipeline.generator, "name", None),
        "model": done.get("model"),
        "text": text,
        "segments": segments,
        "cited": cited,
        "orphan_cites": orphan_cites,
        "cited_files": [hits[i].chunk.source for i in cited],
        "source_texts": {i: hits[i].chunk.text for i in cited},
        "stop_reason": done.get("stop_reason"),
        "markers": done.get("markers"),
        "invalid_markers": done.get("invalid_markers"),
        "latency_ms": latency_ms,
        "retrieval_ms": gate.get("retrieval_ms"),
        "generation_ms": done.get("generation_ms"),
        "gate": gate,
        "model_declined": declined,
    }


# ---- deterministic metrics --------------------------------------------------------------------------

def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile: the ceil(p/100*n)-th smallest value. None for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def _is_answered(it: dict) -> bool:
    return not it["abstained"] and not it.get("error")


def _ratio(num: float, den: float) -> float | None:
    return num / den if den else None


def _is_empty(item: dict) -> bool:
    return not any(c.isalnum() for c in item.get("text") or "")


def aggregate(items: list[dict]) -> dict:
    """Metrics over run_item dicts (optionally with a `judge` entry from judge_item).

    Over the answerable items:
      n, answered (not abstained, no error), abstained, errors, model_declined (heuristic) are counts.
      empty_answers       answered items whose text has no letter or digit (e.g. the model wrote only "[6][1]").
      truncated           answered items whose stop_reason is "length" (the output token cap cut the answer).
      orphan_cites        cite events with no text of their own in answered items (see run_item).
      citation_coverage   answered items with >= 1 valid citation and some text / answered; an empty answer
                          with markers is not "cited". None if nothing was answered.
      citation_precision  micro: all cited chunks of answered items pooled, the share whose source file is in
                          that item's `relevant`; None if there are no citations.
      citation_precision_macro  mean of each answer's own precision, over answers with >= 1 citation.
      invalid_marker_rate sum(invalid_markers) / sum(markers) over items whose generator reports markers; None
                          if none does or there are no markers at all.
    Over the answered answerable items (where the generator actually ran; abstentions take milliseconds and
    would hide its cost):
      latency_p50_ms, latency_p95_ms  nearest-rank percentiles of wall time of retrieve + generate.
    Only when some item was judged:
      faithfulness        supported claims / all claims, pooled over judged segments (micro).
      faithfulness_macro  mean over answers of each answer's supported share.
      relevance_mean      mean relevance score (1-3) over items that have one.
      claims              total judged claims.
      judge_calls, judge_errors  judge calls made and how many ended as judge_error (segments + relevance).
      segments_total, uncited_segments  all segments of answered items, and those without a citation (not judged).
    `unanswerable` is a separate block: n, abstained (the gate refused), answered.
    """
    answerable = [it for it in items if it["answerable"]]
    answered = [it for it in answerable if _is_answered(it)]
    out: dict = {
        "n": len(answerable),
        "answered": len(answered),
        "abstained": sum(it["abstained"] for it in answerable),
        "errors": sum(bool(it.get("error")) for it in answerable),
        "model_declined": sum(it["model_declined"] for it in answered),
        "empty_answers": sum(_is_empty(it) for it in answered),
        "truncated": sum(it.get("stop_reason") == "length" for it in answered),
        "orphan_cites": sum(it.get("orphan_cites", 0) for it in answered),
    }
    with_cites = [it for it in answered if it["cited"] and not _is_empty(it)]
    out["citation_coverage"] = _ratio(len(with_cites), len(answered))
    good = total = 0
    per_answer = []
    for it in with_cites:
        rel = set(it["relevant"])
        g = sum(f in rel for f in it["cited_files"])
        good, total = good + g, total + len(it["cited_files"])
        per_answer.append(g / len(it["cited_files"]))
    out["citation_precision"] = _ratio(good, total)
    out["citation_precision_macro"] = sum(per_answer) / len(per_answer) if per_answer else None
    reporting = [it for it in answerable if it.get("markers") is not None]
    out["invalid_marker_rate"] = _ratio(sum(it["invalid_markers"] or 0 for it in reporting),
                                        sum(it["markers"] for it in reporting))
    times = [it["latency_ms"] for it in answered]
    out["latency_p50_ms"] = percentile(times, 50)
    out["latency_p95_ms"] = percentile(times, 95)
    out.update(_judge_aggregate(answered))
    unanswerable = [it for it in items if not it["answerable"]]
    out["unanswerable"] = {
        "n": len(unanswerable),
        "abstained": sum(it["abstained"] for it in unanswerable),
        "answered": sum(_is_answered(it) for it in unanswerable),
    }
    return out


def _judge_aggregate(answered: list[dict]) -> dict:
    judged = [it for it in answered if "judge" in it]
    if not judged:
        return {}
    supported = claims = calls = errors = 0
    per_answer: list[float] = []
    relevance: list[int] = []
    for it in judged:
        s = c = 0
        for seg in it["judge"]["segments"]:
            calls += 1
            if "judge_error" in seg:
                errors += 1
                continue
            s += sum(cl["supported"] for cl in seg["claims"])
            c += len(seg["claims"])
        supported, claims = supported + s, claims + c
        if c:
            per_answer.append(s / c)
        rel = it["judge"]["relevance"]
        calls += 1
        if "judge_error" in rel:
            errors += 1
        else:
            relevance.append(rel["score"])
    return {
        "faithfulness": _ratio(supported, claims),
        "faithfulness_macro": sum(per_answer) / len(per_answer) if per_answer else None,
        "relevance_mean": sum(relevance) / len(relevance) if relevance else None,
        "claims": claims,
        "judge_calls": calls,
        "judge_errors": errors,
        "segments_total": sum(len(it["segments"]) for it in answered),
        "uncited_segments": sum(1 for it in answered for seg in it["segments"] if not seg["hits"]),
    }


def summarise(items: list[dict], gate: dict) -> dict:
    """Overall + per-slice aggregates for one generator, in the shape written to the results file."""
    overall = aggregate(items)
    unanswerable = overall.pop("unanswerable")
    slices: dict[str, dict] = {}
    for name in sorted({it["slice"] for it in items}):
        if not any(it["slice"] == name and it["answerable"] for it in items):
            continue  # an all-unanswerable slice has no answer metrics
        s = aggregate([it for it in items if it["slice"] == name])
        s.pop("unanswerable")
        slices[name] = s
    return {"overall": overall, "slices": slices, "unanswerable": unanswerable, "gate": gate}


# ---- judge ------------------------------------------------------------------------------------------

class JudgeUnavailable(Exception):
    """The judge model can't be reached or isn't installed. Not retried: it aborts the run."""


class JudgeTimeout(Exception):
    """One judge call timed out. Becomes that call's judge_error; the run goes on."""


class JudgeParseError(ValueError):
    """The judge replied, but not in the required shape. The reason is sent back for one retry."""


# Judge-prompt delimiters inside untrusted text are broken the same way generation._neutralise breaks <source.
_JUDGE_TAG = re.compile(r"<(?=\s*/?\s*(?:sources?|statement|question|answer)\b)", re.IGNORECASE)


def neutralise(s: str) -> str:
    """Break `<source`, `</sources`, `<statement`, ... in text that goes into a judge prompt."""
    return _JUDGE_TAG.sub("‹", s)


MAX_CLAIMS = 6


def parse_faithfulness(raw: str) -> list[dict]:
    """Validate a faithfulness reply: a non-empty list of {"claim": non-blank str, "supported": bool}. The prompt
    asks for at most 6 claims; a longer list keeps the first 6 so one chatty reply doesn't skew the pooled ratio."""
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        raise JudgeParseError("the reply is not valid JSON") from None
    claims = data.get("claims") if isinstance(data, dict) else None
    if not isinstance(claims, list) or not claims:
        raise JudgeParseError('"claims" must be a non-empty list')
    out = []
    for c in claims:
        if (not isinstance(c, dict) or not isinstance(c.get("claim"), str) or not c["claim"].strip()
                or not isinstance(c.get("supported"), bool)):
            raise JudgeParseError('each claim must be {"claim": string, "supported": true or false}')
        out.append({"claim": c["claim"], "supported": c["supported"]})
    return out[:MAX_CLAIMS]


def parse_relevance(raw: str) -> dict:
    """Validate a relevance reply: a verdict from RELEVANCE_SCORES and a string reason. Returns the verdict, its
    1-3 score (answers=3, partial=2, off_topic=1) and the reason."""
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        raise JudgeParseError("the reply is not valid JSON") from None
    if not isinstance(data, dict):
        raise JudgeParseError("the reply must be a JSON object")
    verdict = data.get("verdict")
    if not isinstance(verdict, str) or verdict not in RELEVANCE_SCORES:
        raise JudgeParseError('"verdict" must be one of "answers", "partial", "off_topic"')
    reason = data.get("reason", "")
    if not isinstance(reason, str):
        raise JudgeParseError('"reason" must be a string')
    return {"score": RELEVANCE_SCORES[verdict], "verdict": verdict, "reason": reason}


JUDGE_MAX_TOKENS = 512


class OllamaJudge:
    """Judge backed by a local Ollama model (non-streaming /api/chat with a JSON schema as `format`).

    `chat` is injectable for tests: (messages, schema) -> assistant content."""

    def __init__(self, *, model: str, url: str = "http://localhost:11434", timeout: float = 180.0,
                 chat: Callable[[list[dict], dict], str] | None = None):
        self.model = model
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.chat = chat or self._http_chat

    def _http_chat(self, messages: list[dict], schema: dict) -> str:
        body = json.dumps({
            "model": self.model, "messages": messages, "stream": False, "format": schema,
            # num_predict: constrained JSON decoding can loop (measured: one call ran past 10,000 tokens); a
            # verdict with six claims is ~300 tokens, so a capped, truncated reply fails validation and is retried.
            "options": {"temperature": 0, "num_ctx": 8192, "num_predict": JUDGE_MAX_TOKENS},
        }).encode("utf-8")
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # local: never via HTTP_PROXY
        req = urllib.request.Request(f"{self.url}/api/chat", data=body, method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with opener.open(req, timeout=self.timeout) as resp:
                payload = resp.read()
        except urllib.error.HTTPError as e:
            raise JudgeUnavailable(self._http_message(e)) from None
        except TimeoutError:
            raise JudgeTimeout(f"Ollama didn't respond within {self.timeout:.0f}s.") from None
        except (ValueError, http.client.HTTPException) as e:
            raise JudgeUnavailable(f"Couldn't talk to Ollama at {self.url}: check `ollama_url` in citerag.toml "
                                   f"({type(e).__name__}).") from None
        except (urllib.error.URLError, OSError) as e:
            if isinstance(getattr(e, "reason", None), TimeoutError):
                raise JudgeTimeout(f"Ollama didn't respond within {self.timeout:.0f}s.") from None
            raise JudgeUnavailable(f"Could not reach Ollama at {self.url}. Start it with `ollama serve`, "
                                   f"then pull a model with `ollama pull {self.model}`.") from None
        try:
            data = json.loads(payload)
            content = data["message"]["content"]
        except (ValueError, KeyError, TypeError):
            return ""  # unusable reply: handled as a parse failure (and retried once)
        return content if isinstance(content, str) else ""

    def _http_message(self, e: urllib.error.HTTPError) -> str:
        try:
            detail = e.read().decode("utf-8", "replace").strip()
        except (OSError, http.client.HTTPException):
            detail = ""
        finally:
            e.close()
        try:
            parsed = json.loads(detail)
            if isinstance(parsed, dict) and "error" in parsed:
                detail = str(parsed["error"])
        except ValueError:
            pass
        if e.code == 404 and "not found" in detail.lower() and self.model.lower() in detail.lower():
            return f"Ollama model '{self.model}' isn't installed. Run `ollama pull {self.model}`."
        if e.code == 404:
            return f"Ollama error 404 at {self.url}: {detail or 'no details'}. Is `ollama_url` right?"
        return f"Ollama error {e.code}: {detail}" if detail else f"Ollama error {e.code}"

    def _ask(self, messages: list[dict], schema: dict, parse: Callable[[str], object]) -> dict:
        """One call plus one retry that shows the model its bad reply and why it failed. Never raises on a bad
        reply or a timed-out call: returns {"judge_error", "raw": [replies so far]} so the item is kept and
        visible. Only an unreachable server or a missing model raises (JudgeUnavailable)."""
        try:
            first = self.chat(messages, schema)
        except JudgeTimeout:
            return {"judge_error": "timeout", "raw": []}
        try:
            return {"value": parse(first)}
        except JudgeParseError as e:
            reason = str(e)
        retry = [*messages, {"role": "assistant", "content": first},
                 {"role": "user", "content": RETRY_MESSAGE.format(error=reason)}]
        try:
            second = self.chat(retry, schema)
        except JudgeTimeout:
            return {"judge_error": "timeout", "raw": [first]}
        try:
            return {"value": parse(second)}
        except JudgeParseError as e:
            return {"judge_error": str(e), "raw": [first, second]}

    def faithfulness(self, statement: str, sources: list[str]) -> dict:
        """{"claims": [{"claim", "supported"}]} or {"judge_error", "raw"}."""
        blocks = "\n".join(f'<source n="{i}">{neutralise(s)}</source>' for i, s in enumerate(sources, 1))
        messages = [
            {"role": "system", "content": FAITHFULNESS_SYSTEM},
            {"role": "user", "content": FAITHFULNESS_USER.format(sources=blocks, statement=neutralise(statement))},
        ]
        res = self._ask(messages, FAITHFULNESS_SCHEMA, parse_faithfulness)
        return {"claims": res["value"]} if "value" in res else res

    def relevance(self, question: str, answer: str) -> dict:
        """{"score", "reason"} or {"judge_error", "raw"}."""
        messages = [
            {"role": "system", "content": RELEVANCE_SYSTEM},
            {"role": "user", "content": RELEVANCE_USER.format(question=neutralise(question),
                                                              answer=neutralise(answer))},
        ]
        res = self._ask(messages, RELEVANCE_SCHEMA, parse_relevance)
        return res["value"] if "value" in res else res


def judge_item(judge, item: dict) -> None:
    """Judge an answered item in place. Faithfulness per cited segment against exactly its cited chunks;
    relevance on the whole answer (without the extractive header, which is a label). Uncited segments are not
    judged. Abstained/errored items are left alone."""
    if not _is_answered(item):
        return
    segments = []
    for seg in item["segments"]:
        if not seg["hits"]:
            continue
        res = judge.faithfulness(seg["text"], [item["source_texts"][i] for i in seg["hits"]])
        segments.append({"text": seg["text"], "hits": seg["hits"], **res})
    item["judge"] = {"segments": segments, "relevance": judge.relevance(item["question"], item["text"].replace(EXTRACTIVE_HEADER, ""))}


_EXPECTS = ("unfaithful", "irrelevant", "good")


def load_sanity(path: str | Path) -> list[dict]:
    """Read and validate the sanity fixture before any judge call. Raises ValueError with a readable message."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as e:
        raise ValueError(f"can't read judge sanity file {path}: {e.strerror or e}") from None
    rows = []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            ok = (isinstance(row, dict) and row.get("expect") in _EXPECTS and isinstance(row.get("question"), str)
                  and isinstance(row.get("segments"), list) and row["segments"]
                  and all(isinstance(s, dict) and isinstance(s.get("text"), str) and isinstance(s.get("sources"), list)
                          and all(isinstance(src, dict) and isinstance(src.get("text"), str) for src in s["sources"])
                          for s in row["segments"]))
        except ValueError:
            ok = False
        if not ok:
            raise ValueError(f"malformed judge sanity file {path}, line {n}")
        rows.append(row)
    if not rows:
        raise ValueError(f"judge sanity file {path} has no rows")
    return rows


def judge_sanity(judge, path: str | Path = "eval/judge_sanity.jsonl") -> dict:
    """Run the judge on hand-written bad and good answers and count what it catches.

    Row `expect`: "unfaithful" is caught if any claim in any segment is judged unsupported; "irrelevant" is caught
    if relevance == 1; "good" is a false alarm if any claim is unsupported or relevance == 1. A row where a judge
    call ended as judge_error counts in `judge_errors` and is never counted as caught."""
    rows = load_sanity(path)
    details = []
    bad = caught = good = false_alarms = errors = 0
    for row in rows:
        segs = [judge.faithfulness(s["text"], [src["text"] for src in s["sources"]]) for s in row["segments"]]
        rel = judge.relevance(row["question"], " ".join(s["text"] for s in row["segments"]))
        errored = "judge_error" in rel or any("judge_error" in s for s in segs)
        unsupported = any(not c["supported"] for s in segs for c in s.get("claims", []))
        irrelevant = rel.get("score") == 1
        if row["expect"] == "unfaithful":
            hit = unsupported
        elif row["expect"] == "irrelevant":
            hit = irrelevant
        else:
            hit = unsupported or irrelevant
        hit = hit and not errored  # an errored row is never caught nor a false alarm
        bad += row["expect"] != "good"
        good += row["expect"] == "good"
        caught += hit and row["expect"] != "good"
        false_alarms += hit and row["expect"] == "good"
        errors += errored
        details.append({"id": row.get("id"), "expect": row["expect"], "flagged": hit, "judge_error": errored,
                        "unsupported_claim": unsupported, "relevance": rel.get("score")})
    return {"bad": bad, "caught": caught, "good": good, "false_alarms": false_alarms, "judge_errors": errors,
            "rows": details}


# ---- Ollama preflight -------------------------------------------------------------------------------

def check_ollama(url: str, models: list[str]) -> str | None:
    """Ask Ollama which models are installed. None when it is reachable and has every model, else a message."""
    url = url.rstrip("/")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"{url}/api/tags", timeout=10) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, OSError, ValueError, http.client.HTTPException):
        return f"Could not reach Ollama at {url}. Start it with `ollama serve`."
    models_list = (data.get("models") or []) if isinstance(data, dict) else []
    names = {m.get("name") for m in models_list if isinstance(m, dict)} if isinstance(models_list, list) else set()
    for model in dict.fromkeys(models):
        if model not in names and f"{model}:latest" not in names:
            return f"Ollama model '{model}' isn't installed. Run `ollama pull {model}`."
    return None


# ---- output -----------------------------------------------------------------------------------------

def _fmt_ratio(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def _fmt_ms(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.0f}"


def format_answer_table(results: dict[str, dict]) -> str:
    """Markdown table like evaluation.format_table: `all (n)` per generator first, then per slice. `results`
    maps generator name -> {"overall": aggregate, "slices": {name: aggregate}}. Judge columns appear only if any
    generator was judged."""
    judged = any("faithfulness" in r["overall"] for r in results.values())
    head = ["questions", "generator", "answered", "cited", "cit. precision", "invalid markers"]
    if judged:
        head += ["faithfulness", "relevance"]
    head += ["p50 ms", "p95 ms"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]

    def row(label: str, gen: str, a: dict) -> str:
        cells = [label, gen, f"{a['answered']}/{a['n']}", _fmt_ratio(a["citation_coverage"]),
                 _fmt_ratio(a["citation_precision"]), _fmt_ratio(a["invalid_marker_rate"])]
        if judged:
            rel = a.get("relevance_mean")
            cells += [_fmt_ratio(a.get("faithfulness")), "n/a" if rel is None else f"{rel:.2f}"]
        cells += [_fmt_ms(a["latency_p50_ms"]), _fmt_ms(a["latency_p95_ms"])]
        return "| " + " | ".join(cells) + " |"

    for gen, r in results.items():
        lines.append(row(f"all ({r['overall']['n']})", gen, r["overall"]))
    names = sorted({s for r in results.values() for s in r["slices"]})
    if len(names) > 1:
        for name in names:
            for gen, r in results.items():
                s = r["slices"].get(name)
                if s and s["n"]:  # a slice with only unanswerable questions has no answer metrics
                    lines.append(row(f"{name} ({s['n']})", gen, s))
    return "\n".join(lines)


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def git_dirty() -> bool | None:
    """Whether the working tree has uncommitted changes (`git status --porcelain`); None if git fails."""
    try:
        out = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(out.stdout.strip()) if out.returncode == 0 else None


def default_results_path() -> Path:
    """eval/results/answers-DATE.json; if it exists, -2, -3, ... so an earlier run is never overwritten."""
    base = f"answers-{date.today().isoformat()}"
    path = Path("eval/results") / f"{base}.json"
    n = 2
    while path.exists():
        path = Path("eval/results") / f"{base}-{n}.json"
        n += 1
    return path


def write_results(path: str | Path, payload: dict) -> Path:
    """Write the results JSON (indent 2, non-ASCII kept), creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return path


def judge_info(model: str) -> dict:
    return {"model": model, "prompt_version": JUDGE_PROMPT_VERSION, "max_tokens": JUDGE_MAX_TOKENS}
