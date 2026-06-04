"""
run_classification_tests.py

Interactive test runner for the labelled question fixtures in test/data.

Each fixture is a JSON list of {"question", ...}. Each item declares which
check(s) to run via its key(s):

  - "stance":   expected classification label, e.g.
                "info_seeker - na" / "affected_party - na" /
                "harm_to_others - planning" / "harm_to_others - completed".
                (Older fixtures used "expected_answer" — read as a fallback.)
  - "rejected": bool — whether the answer is expected to be Rejected.

If both keys are present, both must pass.

The script runs the FULL agent workflow on each question and reads the
single `query_status` SSE event, which carries `stance`,
`harm_to_others_tense` and `validate_response_result` (see the
"query_status payload" section in README.MD), then:

  - "stance"   -> PASS iff f"{stance} - {tense}" == the fixture stance
  - "rejected" -> PASS iff (validate_response_result == "Rejected") == rejected

We consume the whole stream: `query_status` carries the classification,
and the `answer` / `references` events that follow it give us the agent's
actual answer text and its source list (one reference per line in the
Excel cell).

For each fixture chosen, an Excel file with the SAME basename is written
next to it (e.g. unanswered_questions.json -> unanswered_questions.xlsx).

Usage (Windows PowerShell, from repo root):

    $env:PYTHONPATH="."; python -u test/run_classification_tests.py

The script asks at the start which fixture(s) to run. You can also pass
fixtures and config on the command line to skip the prompt:

    python test/run_classification_tests.py `
        --files unanswered_questions.json,harm_to_others-planning_questions.json `
        --vector-index hvaerinnafor_unified --psa-ssa-threshold 0
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from dotenv import find_dotenv, load_dotenv
from langchain_openai import AzureChatOpenAI
from llama_index.core import StorageContext, load_index_from_storage

from config import VECTOR_INDEX_MAP, ServerSettings, VectorIndexStore
from answer_utils import get_answer_as_stream
from query_utils import QuerySettings

# UTF-8 stdout so Norwegian characters print cleanly on Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass


load_dotenv(find_dotenv(), override=True)


# --- Singletons reused by get_answer_as_stream ---------------------------
vector_store = VectorIndexStore()
server_settings = ServerSettings()

LLM = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    timeout=120,
    temperature=0.0,
    verbose=False,
)
server_settings.set_llm(LLM)

# This script sits in test/. Fixtures (JSON) are read from test/data/;
# Excel output (per-fixture + summary) is written to test/results/.
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_TEST_DIR, "data")
RESULTS_DIR = os.path.join(_TEST_DIR, "results")


def _results_xlsx_for(fixture_path: str) -> str:
    """Output .xlsx path in test/results/ for a given fixture JSON path."""
    base = os.path.splitext(os.path.basename(fixture_path))[0] + ".xlsx"
    return os.path.join(RESULTS_DIR, base)


def _read_all_indexes_from_storage(vector_map) -> bool:
    """Load every index in VECTOR_INDEX_MAP into the module vector_store."""
    found_any = False
    for item in vector_map:
        start = time.time()
        name, storage, desc = item["name"], item["storage"], item["description"]
        if os.path.exists(storage):
            logging.info("Loading index '%s' from %s", name, storage)
            ctx = StorageContext.from_defaults(persist_dir=storage)
            vector_store.add(name, load_index_from_storage(ctx), desc)
            found_any = True
        else:
            logging.warning("Index directory not found: %s", storage)
        logging.info("Loaded %s in %.2fs", name, time.time() - start)
    vector_store.indexes_loaded = found_any
    return found_any


# --- Fixture discovery & selection ---------------------------------------


def discover_fixtures() -> List[str]:
    """Return labelled question fixtures in DATA_DIR (sorted).

    Matches '*_questions.json' but skips the '*_log.json' traces written by
    the find_* generators.
    """
    out = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*_questions.json"))):
        if path.endswith("_log.json"):
            continue
        out.append(path)
    return out


def choose_fixtures(fixtures: List[str]) -> List[str]:
    """Interactive menu: pick one, several (comma list), or all."""
    print("\nAvailable test fixtures:")
    for i, path in enumerate(fixtures, 1):
        print(f"  {i}. {os.path.basename(path)}")
    print("  a. all of the above")

    while True:
        raw = input(
            "\nWhich test(s) to run? (number, comma-separated numbers, or 'a'): "
        ).strip().lower()
        if not raw:
            continue
        if raw in ("a", "all"):
            return fixtures
        try:
            picks = [int(x) for x in raw.replace(" ", "").split(",") if x]
            chosen = [fixtures[i - 1] for i in picks if 1 <= i <= len(fixtures)]
            if chosen:
                return chosen
        except (ValueError, IndexError):
            pass
        print("  Invalid choice — try again (e.g. '1', '1,3', or 'a').")


# --- Question loading -----------------------------------------------------


def load_questions(path: str) -> List[Dict[str, Any]]:
    """Load a fixture, preserving its test keys per item.

    Each item is ``{"question", ...}`` plus whichever check key(s) it carries:
      - ``stance``   — expected classification label (e.g. "info_seeker - na").
                       Older fixtures used ``expected_answer`` (read as fallback).
      - ``rejected`` — bool: whether the answer is expected to be Rejected.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list")
    out: List[Dict[str, Any]] = []
    for item in data:
        if isinstance(item, dict) and "question" in item:
            entry: Dict[str, Any] = {"question": str(item["question"]).strip()}
            stance = item.get("stance", item.get("expected_answer"))
            if stance is not None and str(stance).strip():
                entry["stance"] = str(stance).strip()
            if "rejected" in item:
                entry["rejected"] = bool(item["rejected"])
            out.append(entry)
        elif isinstance(item, str):
            out.append({"question": item.strip()})
    return out


# --- Per-question run -----------------------------------------------------

# Reference bullets arrive as "[Name](url) ||IMG|| icon_url" or "[Name](url)"
# — see emit_query_answer_references in agent_workflow_answer.py.
_REF_LINE_RE = re.compile(
    r"^\[(?P<name>.+?)\]\((?P<url>.+?)\)(?:\s*\|\|IMG\|\|\s*(?P<icon>.+?))?\s*$"
)


def _format_references(lines: List[str]) -> str:
    """Parse streamed `references` event lines into one 'Title — URL' per row.

    Returns a single string with a newline between each reference so the
    whole list lands in one Excel cell.
    """
    out: List[str] = []
    seen: set[str] = set()
    for raw in lines:
        for piece in raw.splitlines():
            piece = piece.strip()
            if not piece:
                continue
            m = _REF_LINE_RE.match(piece)
            if not m:
                continue
            url = m.group("url").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            out.append(f"{m.group('name').strip()} — {url}")
    return "\n".join(out)


async def classify_via_agent(
    question: str,
    vector_index: str,
    psa_ssa_threshold: float,
    similarity_top_k: int,
    similarity_cutoff: float,
    qa_bank_index: Optional[str],
    claims_valid_threshold: float,
    entailment_check: bool,
) -> Dict[str, Any]:
    """Run the full agent and return classification + answer + references.

    Consumes the entire stream: the `query_status` event carries the
    classification (stance, harm_to_others_tense, validate_response_result,
    query_severity, refined_query); the `answer` and `references` events that
    follow give the agent's answer text and its source list. The returned
    dict is the query_status payload plus `answer` and `references` keys.
    """
    qs = QuerySettings(
        user_content=question,
        vectorIndex=vector_index,
        response_mode="tree_summarize",
        similarity_top_k=similarity_top_k,
        similarity_cutoff=similarity_cutoff,
        psa_ssa_threshold=psa_ssa_threshold,
        qa_bank_index=qa_bank_index,
        claims_valid_threshold=claims_valid_threshold,
        entailment_check=entailment_check,
        debug_emit_nodes=True,  # ask the agent to emit its exact retrieved nodes
    )

    status: Dict[str, Any] = {}
    answer_buf: List[str] = []
    ref_lines: List[str] = []
    node_dump: List[Dict[str, Any]] = []
    sysinfo_lines: List[str] = []
    try:
        async for chunk in get_answer_as_stream(qs, server_settings, vector_store):
            if not isinstance(chunk, dict):
                continue
            event = chunk.get("event")
            text = None
            for k in ("structured_answer_delta", "delta", "text", "message", "content"):
                v = chunk.get(k)
                if isinstance(v, str):
                    text = v
                    break

            if event == "query_status" and text:
                try:
                    status = json.loads(text)
                except json.JSONDecodeError:
                    pass
            elif event == "answer" and text:
                answer_buf.append(text)
            elif event == "references" and text:
                ref_lines.append(text)
            elif event == "retrieved_nodes" and text:
                try:
                    node_dump = json.loads(text)
                except json.JSONDecodeError:
                    pass
            elif event == "systeminfo" and text:
                sysinfo_lines.append(text)
    except Exception as e:  # noqa: BLE001 - record any pipeline failure per-row
        return {"error": f"{type(e).__name__}: {e}"}

    status["answer"] = "".join(answer_buf).strip()
    status["references"] = _format_references(ref_lines)
    status["nodes_text"] = _format_node_dump(node_dump)
    status["systeminfo"] = "\n".join(sysinfo_lines)
    return status


def _format_node_dump(node_dump: List[Dict[str, Any]]) -> str:
    """Concatenate the agent's exact retrieved nodes into one cell.

    `node_dump` is the JSON from the `retrieved_nodes` SSE event — the precise
    set query_grounded retrieved for this answer, each with score/url/text.
    """
    parts: List[str] = []
    for n in node_dump or []:
        score = n.get("score")
        tag = f"[score={score:.3f}] " if isinstance(score, (int, float)) else ""
        url = n.get("url") or ""
        ntype = n.get("node_type") or ""
        head = f"{tag}({ntype}) {url}".rstrip()
        parts.append(f"{head}\n{(n.get('text') or '').strip()}")
    return "\n\n---\n\n".join(parts)


def predicted_label(status: Dict[str, Any]) -> str:
    """Build the 'stance - tense' label the fixtures use as expected_answer."""
    stance = status.get("stance", "") or ""
    tense = status.get("harm_to_others_tense", "na") or "na"
    if not stance:
        return ""
    return f"{stance} - {tense}"


def _expected_display(item: Dict[str, Any]) -> str:
    """Human-readable target(s) for the 'expected' Excel column / console."""
    parts: List[str] = []
    if "stance" in item:
        parts.append(str(item["stance"]))
    if "rejected" in item:
        parts.append(f"rejected={item['rejected']}")
    return ", ".join(parts)


def verdict(item: Dict[str, Any], status: Dict[str, Any]) -> str:
    """PASS / FAIL / ERROR for one fixture item.

    Runs whichever check(s) the item declares — if both keys are present, all
    must pass:
      - ``rejected``: the answer's Rejected/Accepted state must match.
      - ``stance``: the predicted "stance - tense" label must match.
    """
    if status.get("error"):
        return "ERROR"

    checks: List[bool] = []
    stance = item.get("stance", "")

    # Rejection test.
    if "rejected" in item:
        want_rejected = bool(item["rejected"])
        got_rejected = status.get("validate_response_result") == "Rejected"
        checks.append(got_rejected == want_rejected)

    # Stance/classification test.
    if stance:
        checks.append(predicted_label(status) == stance)

    if not checks:
        return "ERROR"
    return "PASS" if all(checks) else "FAIL"


async def run_fixture(
    path: str,
    vector_index: str,
    psa_ssa_threshold: float,
    similarity_top_k: int,
    similarity_cutoff: float,
    qa_bank_index: Optional[str],
    claims_valid_threshold: float,
    entailment_check: bool,
) -> Dict[str, Any]:
    """Run every question in *path*, print progress, write the Excel.

    Returns a per-fixture stats dict used to build the overall summary.
    """
    questions = load_questions(path)
    name = os.path.basename(path)
    run_date = datetime.now(ZoneInfo("Europe/Oslo")).strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n=== {name} — {len(questions)} questions ===")

    rows: List[Dict[str, Any]] = []
    n_pass = n_fail = n_err = 0

    # Exact call parameters — recorded on failing rows so a FAIL is reproducible.
    agent_params = (
        f"vector_index={vector_index}, psa_ssa_threshold={psa_ssa_threshold}, "
        f"similarity_top_k={similarity_top_k}, similarity_cutoff={similarity_cutoff}, "
        f"qa_bank_index={qa_bank_index}, claims_valid_threshold={claims_valid_threshold}, "
        f"entailment_check={entailment_check}"
    )

    for i, item in enumerate(questions, 1):
        q = item["question"]
        expected = _expected_display(item)
        short = q[:80].replace("\n", " ")
        print(f"[{i}/{len(questions)}] {short}", flush=True)

        status = await classify_via_agent(
            q,
            vector_index=vector_index,
            psa_ssa_threshold=psa_ssa_threshold,
            similarity_top_k=similarity_top_k,
            similarity_cutoff=similarity_cutoff,
            qa_bank_index=qa_bank_index,
            claims_valid_threshold=claims_valid_threshold,
            entailment_check=entailment_check,
        )

        v = verdict(item, status)
        if v == "PASS":
            n_pass += 1
        elif v == "FAIL":
            n_fail += 1
        else:
            n_err += 1

        pred = predicted_label(status)
        detail = pred or "(no stance)"
        print(
            f"      -> {v}   expected={expected!r}  "
            f"predicted={detail!r}  validate={status.get('validate_response_result', '')!r}",
            flush=True,
        )

        # The exact node set the agent retrieved for this answer (emitted via
        # the `retrieved_nodes` event). Recorded for every row, pass or fail.
        nodes_text = status.get("nodes_text", "")
        rows.append(
            {
                "question": q,
                "expected": expected,
                "result": v,
                "predicted_label": pred,
                "predicted_stance": status.get("stance", ""),
                "predicted_tense": status.get("harm_to_others_tense", ""),
                "validate_response_result": status.get("validate_response_result", ""),
                "query_severity": status.get("query_severity", ""),
                # Always recorded, for both passing and failing rows.
                "relevancy_score": status.get("best_node_score", ""),
                "agent_params": agent_params,
                "nodes_text": nodes_text,
                "systeminfo": status.get("systeminfo", ""),
                "answer": status.get("answer", ""),
                "references": status.get("references", ""),
                "refined_query": status.get("refined_query", ""),
                "error": status.get("error", ""),
                "run_date": run_date,
            }
        )

    total = len(questions)
    accuracy = (n_pass / total * 100.0) if total else 0.0
    print(
        f"--- {name}: PASS={n_pass}  FAIL={n_fail}  ERROR={n_err}  "
        f"accuracy={accuracy:.1f}%"
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_xlsx = _results_xlsx_for(path)
    df = pd.DataFrame(
        rows,
        columns=[
            "run_date",
            "question",
            "refined_query",
            "expected",
            "result",
            "predicted_label",
            "predicted_stance",
            "predicted_tense",
            "validate_response_result",
            "query_severity",
            "relevancy_score",
            "agent_params",
            "nodes_text",
            "systeminfo",
            "answer",
            "references",
            "error",
        ],
    )
    df = _strip_illegal_excel_chars(df)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="results")
        _style_sheet(writer.sheets["results"], df.columns.tolist())
    print(f"Wrote {len(df)} rows to {out_xlsx}")

    return {
        "fixture": name,
        "total": total,
        "pass": n_pass,
        "fail": n_fail,
        "error": n_err,
        "pass_pct": round(accuracy, 1),
    }


def _strip_illegal_excel_chars(df: pd.DataFrame) -> pd.DataFrame:
    """Remove control characters openpyxl refuses to write to a worksheet.

    LLM-generated answer/nodes text can contain control characters (e.g.
    \\x00-\\x08, \\x0b, \\x0c, \\x0e-\\x1f) that openpyxl rejects with
    IllegalCharacterError. Strip them from every string cell so the export
    never aborts mid-run.
    """
    return df.map(
        lambda v: ILLEGAL_CHARACTERS_RE.sub("", v) if isinstance(v, str) else v
    )


def _style_sheet(ws, columns: List[str]) -> None:
    """Widen long-text columns and wrap text so newlines render in Excel.

    Without wrap_text the '\\n' between references sits in the cell value but
    Excel shows it on a single line — wrapping makes one reference per line.
    """
    from openpyxl.utils import get_column_letter
    from openpyxl.styles import Alignment

    widths = {
        "question": 60,
        "answer": 80,
        "references": 70,
        "refined_query": 50,
        "agent_params": 50,
        "nodes_text": 90,
        "systeminfo": 90,
    }
    wrap_cols = {
        "question", "answer", "references", "refined_query",
        "expected", "agent_params", "nodes_text", "systeminfo",
    }
    top_wrap = Alignment(wrap_text=True, vertical="top")
    for idx, col in enumerate(columns, start=1):
        letter = get_column_letter(idx)
        ws.column_dimensions[letter].width = widths.get(col, 18)
        if col in wrap_cols:
            for cell in ws[letter][1:]:  # skip header row
                cell.alignment = top_wrap


SUMMARY_XLSX = os.path.join(RESULTS_DIR, "summary.xlsx")


def _delete_existing_excels(chosen: List[str]) -> None:
    """Remove each fixture's .xlsx plus the summary, before a fresh run."""
    targets = [_results_xlsx_for(p) for p in chosen]
    targets.append(SUMMARY_XLSX)
    for t in targets:
        if os.path.exists(t):
            try:
                os.remove(t)
                print(f"Deleted existing {os.path.basename(t)}")
            except OSError as e:
                print(f"Could not delete {t}: {e}")


def _write_summary(results: List[Dict[str, Any]]) -> None:
    """Print and write an overall summary (status + pass% per sub-test)."""
    run_date = datetime.now(ZoneInfo("Europe/Oslo")).strftime("%Y-%m-%d %H:%M:%S")

    rows = list(results)
    totals = {
        "fixture": "TOTAL",
        "total": sum(r["total"] for r in results),
        "pass": sum(r["pass"] for r in results),
        "fail": sum(r["fail"] for r in results),
        "error": sum(r["error"] for r in results),
    }
    totals["pass_pct"] = round(totals["pass"] / totals["total"] * 100.0, 1) if totals["total"] else 0.0
    rows.append(totals)

    # Console table
    print("\n================ SUMMARY ================")
    print(f"{'fixture':<42} {'total':>5} {'pass':>5} {'fail':>5} {'err':>4} {'pass%':>6}")
    print("-" * 72)
    for r in rows:
        print(
            f"{r['fixture']:<42} {r['total']:>5} {r['pass']:>5} {r['fail']:>5} "
            f"{r['error']:>4} {r['pass_pct']:>6.1f}"
        )
    print("=" * 72)

    df = pd.DataFrame(rows, columns=["fixture", "total", "pass", "fail", "error", "pass_pct"])
    df["run_date"] = run_date
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_excel(SUMMARY_XLSX, index=False)
    print(f"Wrote summary to {SUMMARY_XLSX}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run labelled question fixtures through the agent and export to Excel."
    )
    ap.add_argument(
        "--files",
        default=None,
        help="Comma-separated fixture filenames (basename or path) to run. "
        "If omitted, the script shows an interactive menu.",
    )
    ap.add_argument("--vector-index", default="hvaerinnafor_unified", help="Main vector index.")
    ap.add_argument(
        "--psa-ssa-threshold",
        type=float,
        default=0.0,
        help="Cascade cutoff. 0 = pure unified mode (matches how these fixtures were generated).",
    )
    ap.add_argument("--topk", type=int, default=5, help="similarity_top_k.")
    ap.add_argument("--cutoff", type=float, default=0.75, help="similarity_cutoff.")
    ap.add_argument("--qa-bank-index", default="hvaerinnafor_qa_bank", help="QA-bank index name.")
    ap.add_argument(
        "--claims-threshold",
        type=float,
        default=1.0,
        help="Min fraction of cited claims that must be supported to keep an answer valid "
        "(agent default 1.0).",
    )
    ap.add_argument(
        "--no-entailment",
        dest="entailment_check",
        action="store_false",
        help="Disable the LLM entailment gate (on by default) to A/B its effect.",
    )
    ap.set_defaults(entailment_check=True)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )

    fixtures = discover_fixtures()
    if not fixtures:
        raise SystemExit(f"No '*_questions.json' fixtures found in {DATA_DIR}")

    # Resolve which fixtures to run (CLI flag wins over the interactive menu).
    if args.files:
        wanted = [f.strip() for f in args.files.split(",") if f.strip()]
        by_base = {os.path.basename(p): p for p in fixtures}
        chosen: List[str] = []
        for w in wanted:
            if w in by_base:
                chosen.append(by_base[w])
            elif os.path.isfile(w):
                chosen.append(os.path.abspath(w))
            else:
                raise SystemExit(f"Fixture not found: {w}  (known: {sorted(by_base)})")
    else:
        chosen = choose_fixtures(fixtures)

    # 1) Delete prior Excel output (per-fixture files + summary) before running.
    _delete_existing_excels(chosen)

    print("\nLoading indexes from storage (this takes ~30-60s)...")
    if not _read_all_indexes_from_storage(VECTOR_INDEX_MAP):
        raise SystemExit("No indexes loaded — check VECTOR_INDEX_MAP storage paths.")
    if vector_store.get(args.vector_index) is None:
        raise SystemExit(
            f"Index '{args.vector_index}' not loaded. Loaded: "
            + ", ".join(e.name for e in vector_store.get_all())
        )

    print(
        f"Running {len(chosen)} fixture(s) with vectorIndex={args.vector_index}, "
        f"psa_ssa_threshold={args.psa_ssa_threshold}, topk={args.topk}, cutoff={args.cutoff}, "
        f"claims_threshold={args.claims_threshold}, entailment_check={args.entailment_check}"
    )

    async def _run_all() -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for path in chosen:
            results.append(
                await run_fixture(
                    path,
                    vector_index=args.vector_index,
                    psa_ssa_threshold=args.psa_ssa_threshold,
                    similarity_top_k=args.topk,
                    similarity_cutoff=args.cutoff,
                    qa_bank_index=args.qa_bank_index,
                    claims_valid_threshold=args.claims_threshold,
                    entailment_check=args.entailment_check,
                )
            )
        return results

    results = asyncio.run(_run_all())

    # 2) Overall summary across all sub-tests (printed + written to Excel).
    if results:
        _write_summary(results)

    print("\nDone.")


if __name__ == "__main__":
    main()
