"""
Wambule Language Extraction Pipeline
=====================================
Two-stage AI pipeline:
  Stage 1: Gemini  → accurate phoneme extraction
  Stage 2: Claude  → structured schema enrichment
  Stage 3: Storage → JSON + CSV + PostgreSQL

Usage:
  # Full pipeline (both stages)
  python pipeline.py --pdf opgenort.pdf --pages 545-947

  # Test mode (5 pages only)
  python pipeline.py --pdf opgenort.pdf --pages 545-550 --test

  # Skip Stage 1, re-enrich existing Gemini output
  python pipeline.py --enrich-only output/wambule_raw_20240101.json

  # With PostgreSQL
  python pipeline.py --pdf opgenort.pdf --pages 545-947 --db
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load .env file
load_dotenv()

from stage1_gemini import run_gemini_stage
from stage2_claude import run_claude_stage
from storage import save_all

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))
OUTPUT_DIR.mkdir(exist_ok=True)


def get_db_config() -> dict | None:
    """Build DB config from environment variables."""
    host = os.getenv("DB_HOST")
    if not host:
        return None
    return {
        "host":     host,
        "port":     os.getenv("DB_PORT", "5432"),
        "dbname":   os.getenv("DB_NAME", "wambule"),
        "user":     os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASS", ""),
    }


def parse_page_range(s: str) -> tuple:
    if "-" in s:
        parts = s.split("-")
        return int(parts[0]), int(parts[1])
    return int(s), int(s)


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║         WAMBULE LANGUAGE EXTRACTION PIPELINE             ║
║         Gemini (Eyes) + Claude (Architect)               ║
╚══════════════════════════════════════════════════════════╝
""")


def print_summary(raw_entries: list, enriched_entries: list, paths: dict):
    total         = len(enriched_entries)
    loan_words    = len([e for e in enriched_entries if e.get("is_loan_word")])
    native_words  = len([e for e in enriched_entries if e.get("is_loan_word") == False])
    with_sub      = len([e for e in enriched_entries if e.get("sub_entries")])
    needs_review  = len([e for e in enriched_entries if e.get("needs_review") or e.get("_enrichment_failed")])
    errors_s1     = len([e for e in raw_entries if e.get("_parse_error") or e.get("_stage1_error")])

    print(f"""
╔══════════════════════════════════════════════════════════╗
║                  PIPELINE COMPLETE                       ║
╠══════════════════════════════════════════════════════════╣
║  Total entries extracted  : {total:<28} ║
║  Loan words               : {loan_words:<28} ║
║  Native Wambule words     : {native_words:<28} ║
║  Entries with sub-entries : {with_sub:<28} ║
║  Flagged for review       : {needs_review:<28} ║
║  Stage 1 errors           : {errors_s1:<28} ║
╠══════════════════════════════════════════════════════════╣
║  OUTPUT FILES                                            ║
║  JSON  : {str(paths.get('json','')):<48} ║
║  CSV   : {str(paths.get('csv','')):<48} ║
╚══════════════════════════════════════════════════════════╝
""")


def run_full_pipeline(
    pdf_path: str,
    start_page: int,
    end_page: int,
    gemini_key: str,
    claude_key: str,
    use_db: bool = False,
    batch_size: int = 5,
    test_mode: bool = False
):
    print_banner()

    if test_mode:
        print("  ⚡ TEST MODE — processing 5 pages only\n")
        end_page = min(start_page + 4, end_page)

    # ── Stage 1: Gemini Extraction ─────────────────────────────
    raw_entries = run_gemini_stage(
        pdf_path=pdf_path,
        api_key=gemini_key,
        start_page=start_page,
        end_page=end_page,
        batch_size=batch_size,
    )

    # Save raw output immediately (so you don't lose Stage 1 if Stage 2 fails)
    raw_path = OUTPUT_DIR / f"wambule_raw_{TIMESTAMP}.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_entries, f, ensure_ascii=False, indent=2)
    print(f"\n  ✓ Raw Stage 1 output saved → {raw_path}")

    # ── Stage 2: Claude Enrichment ─────────────────────────────
    enriched_entries = run_claude_stage(
        raw_entries=raw_entries,
        api_key=claude_key,
        entries_per_batch=20,
    )

    # ── Stage 3: Save All Outputs ───────────────────────────────
    db_config = get_db_config() if use_db else None
    label = f"test_{start_page}-{end_page}" if test_mode else f"full_{start_page}-{end_page}"
    paths = save_all(enriched_entries, db_config=db_config, label=label)

    print_summary(raw_entries, enriched_entries, paths)
    return enriched_entries


def run_enrich_only(raw_json_path: str, claude_key: str, use_db: bool = False):
    """Re-run Stage 2 on existing Stage 1 output."""
    print_banner()
    print(f"  Re-enriching: {raw_json_path}\n")

    with open(raw_json_path, "r", encoding="utf-8") as f:
        raw_entries = json.load(f)

    enriched_entries = run_claude_stage(
        raw_entries=raw_entries,
        api_key=claude_key,
        entries_per_batch=20,
    )

    db_config = get_db_config() if use_db else None
    paths = save_all(enriched_entries, db_config=db_config, label="enriched")
    print_summary(raw_entries, enriched_entries, paths)
    return enriched_entries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wambule Extraction Pipeline")
    parser.add_argument("--pdf",          help="Path to the PDF file")
    parser.add_argument("--pages",        default="545-947", help="Page range e.g. 545-947")
    parser.add_argument("--test",         action="store_true", help="Test mode: 5 pages only")
    parser.add_argument("--db",           action="store_true", help="Save to PostgreSQL")
    parser.add_argument("--enrich-only",  metavar="JSON_FILE", help="Skip Stage 1, re-enrich existing JSON")
    parser.add_argument("--gemini-key",   default=os.getenv("GEMINI_API_KEY"), help="Gemini API key")
    parser.add_argument("--claude-key",   default=os.getenv("ANTHROPIC_API_KEY"), help="Claude API key")
    parser.add_argument("--batch-size",   type=int, default=5, help="Pages per Gemini batch")

    args = parser.parse_args()

    # Validate API keys
    if not args.gemini_key:
        print("ERROR: GEMINI_API_KEY not set. Add to .env or pass --gemini-key")
        sys.exit(1)
    if not args.claude_key:
        print("ERROR: ANTHROPIC_API_KEY not set. Add to .env or pass --claude-key")
        sys.exit(1)

    if args.enrich_only:
        run_enrich_only(args.enrich_only, args.claude_key, use_db=args.db)
    else:
        if not args.pdf:
            print("ERROR: --pdf is required unless using --enrich-only")
            sys.exit(1)
        start, end = parse_page_range(args.pages)
        run_full_pipeline(
            pdf_path=args.pdf,
            start_page=start,
            end_page=end,
            gemini_key=args.gemini_key,
            claude_key=args.claude_key,
            use_db=args.db,
            batch_size=args.batch_size,
            test_mode=args.test,
        )