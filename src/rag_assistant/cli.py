"""Command line interface for RAG Study Assistant."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from rag_assistant import __version__
from rag_assistant.config import load_config
from rag_assistant.services import subject_service
from rag_assistant.logging import configure_logging, get_logger, get_run_id
from rag_assistant.ingest.ocr.selftest import run_ocr_selftest
from rag_assistant.ingest import pipeline


logger = get_logger(__name__)


def _not_implemented_handler(args: argparse.Namespace) -> None:
    message = {"command": args.command, "args": vars(args), "status": "Not implemented yet"}
    print(json.dumps(message, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG Study Assistant CLI")
    parser.add_argument("--version", action="version", version=f"rag-study-assistant {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    def add_stub(name: str, help_text: str, extra_args: List[tuple[str, dict]] | None = None) -> None:
        sub = subparsers.add_parser(name, help=help_text)
        if extra_args:
            for arg, kwargs in extra_args:
                sub.add_argument(arg, **kwargs)
        sub.set_defaults(func=_not_implemented_handler)

    ingest_parser = subparsers.add_parser("ingest", help="Index uploaded assets for a subject")
    ingest_parser.add_argument("--subject", help="Subject id to ingest")
    ingest_parser.add_argument("--all-subjects", action="store_true", help="Ingest all subjects")
    ingest_parser.add_argument("--force", action="store_true", help="Re-run pipeline even if assets are indexed")
    ingest_parser.add_argument("--limit", type=int, help="Maximum assets to process (total across subjects)")
    ingest_parser.set_defaults(func=_ingest_handler)

    add_stub("ask", "Ask a question", [("question", {"help": "Question to ask"})])
    add_stub("summarize", "Summarize content", [("--subject", {"help": "Subject id"})])
    add_stub("flashcards", "Generate flashcards", [("--count", {"type": int, "default": 10})])
    add_stub("quiz", "Generate a quiz", [("--subject", {"help": "Subject id"})])
    add_stub("eval", "Run evaluation", [])

    subjects_parser = subparsers.add_parser("subjects", help="List or create subjects")
    subjects_parser.add_argument("--create", help="Create a new subject with this name")
    subjects_parser.set_defaults(func=_subjects_handler)

    reset_all_parser = subparsers.add_parser("reset-all", help="Reset generated artifacts for all assets")
    reset_all_parser.set_defaults(func=_reset_all_handler)

    doctor_parser = subparsers.add_parser("doctor", help="Run OCR self-test")
    doctor_parser.set_defaults(func=_doctor_handler)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return

    config = load_config()
    configure_logging(config.logging.level)
    logger.info("Starting CLI", extra={"run_id": get_run_id(), "command": args.command, "env": config.app.environment})
    args.func(args)


def _subjects_handler(args: argparse.Namespace) -> None:
    if args.create:
        created = subject_service.create_subject(args.create)
        print(json.dumps({"status": "created", **created}, indent=2))
    subjects = subject_service.list_subjects()
    print(json.dumps({"subjects": subjects}, indent=2))


def _reset_all_handler(args: argparse.Namespace) -> None:
    script_path = Path(__file__).resolve().parent.parent.parent / "scripts" / "reset_all_assets.sh"
    if not script_path.exists():
        print("reset_all_assets.sh not found", file=sys.stderr)
        return
    result = subprocess.run([str(script_path)], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)


def _doctor_handler(args: argparse.Namespace) -> None:
    cfg = load_config()
    try:
        res = run_ocr_selftest(cfg)
        output = {
            "ocr_engine": cfg.ingest.ocr_engine,
            "ocr_lang": cfg.ingest.ocr_lang,
            "tesseract_cmd": cfg.ingest.tesseract_cmd,
            "tessdata_dir": cfg.ingest.tessdata_dir,
            "result": res,
        }
        print(json.dumps(output, indent=2))
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        sys.exit(1)


def _ingest_handler(args: argparse.Namespace) -> None:
    cfg = load_config()
    subject_ids: List[str] = []
    if args.all_subjects:
        subject_ids = [s["subject_id"] for s in subject_service.list_subjects()]
        if not subject_ids:
            print(json.dumps({"error": "No subjects found"}, indent=2))
            return
    elif args.subject:
        subject_ids = [args.subject]
    else:
        print(json.dumps({"error": "Provide --subject or --all-subjects"}, indent=2), file=sys.stderr)
        sys.exit(1)

    remaining: Optional[int] = args.limit
    results = []
    any_failures = False

    for sid in subject_ids:
        subj = subject_service.get_subject(sid)
        if subj is None:
            results.append({"subject_id": sid, "error": "Subject not found"})
            any_failures = True
            continue

        local_limit = remaining if remaining is not None else None
        try:
            summary = pipeline.process_subject_new_assets(sid, cfg, force=args.force, limit=local_limit)
        except Exception as exc:
            results.append({"subject_id": sid, "assets_processed": 0, "assets_indexed": 0, "failures": [{"error": str(exc)}]})
            any_failures = True
            continue

        processed = summary.get("processed", 0)
        indexed = summary.get("indexed", 0)
        failures = [
            {"asset_id": d.get("asset_id"), "error": d.get("error")}
            for d in summary.get("details", [])
            if d.get("stage") == "failed" or d.get("error")
        ]
        any_failures = any_failures or bool(failures)
        results.append(
            {
                "subject_id": sid,
                "assets_processed": processed,
                "assets_indexed": indexed,
                "failures": failures,
            }
        )
        if remaining is not None:
            remaining = max(0, remaining - processed)
            if remaining == 0:
                break

    print(json.dumps({"subjects": results}, indent=2))
    if any_failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
