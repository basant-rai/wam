"""
Storage Layer
=============
Saves enriched entries to JSON, CSV, and PostgreSQL.
"""

import csv
import json
import uuid
import os
from pathlib import Path
from datetime import datetime

try:
    import psycopg2
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))
OUTPUT_DIR.mkdir(exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def assign_ids(entries: list) -> list:
    """Assign UUIDs to entries that don't have one."""
    for e in entries:
        if not e.get("id"):
            e["id"] = str(uuid.uuid4())
    return entries


def save_json(entries: list, label: str = "") -> Path:
    """Save all entries to a pretty-printed JSON file."""
    filename = f"wambule_{label}_{TIMESTAMP}.json" if label else f"wambule_{TIMESTAMP}.json"
    path = OUTPUT_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"  ✓ JSON  → {path} ({len(entries)} entries)")
    return path


def save_csv(entries: list, label: str = "") -> Path:
    """Save flattened entries to CSV for spreadsheet review."""
    filename = f"wambule_{label}_{TIMESTAMP}.csv" if label else f"wambule_{TIMESTAMP}.csv"
    path = OUTPUT_DIR / filename

    fieldnames = [
        "id", "headword", "headword_variants", "phoneme_group",
        "is_loan_word", "native_or_loan", "part_of_speech",
        "definition_en", "definition_ne_script", "definition_ne_translit",
        "scientific_name", "dialect", "domain_tags",
        "loan_sources", "lit_references", "cross_references",
        "sub_entries_count", "source_page", "verified", "needs_review"
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for e in entries:
            defn = e.get("definition", {})
            ne = defn.get("ne", {}) if isinstance(defn, dict) else {}

            writer.writerow({
                "id":                    e.get("id", ""),
                "headword":              e.get("headword", e.get("raw_headword", "")),
                "headword_variants":     "|".join(e.get("headword_variants", [])),
                "phoneme_group":         e.get("phoneme_group", ""),
                "is_loan_word":          e.get("is_loan_word", ""),
                "native_or_loan":        e.get("native_or_loan", ""),
                "part_of_speech":        e.get("part_of_speech", e.get("raw_pos", "")),
                "definition_en":         defn.get("en", "") if isinstance(defn, dict) else str(defn),
                "definition_ne_script":  ne.get("script", "") if isinstance(ne, dict) else "",
                "definition_ne_translit":ne.get("translit", "") if isinstance(ne, dict) else "",
                "scientific_name":       e.get("scientific_name", "") or "",
                "dialect":               e.get("dialect", "Unknown"),
                "domain_tags":           "|".join(e.get("domain_tags", [])),
                "loan_sources":          json.dumps(e.get("loan_sources", []), ensure_ascii=False),
                "lit_references":        "|".join(e.get("lit_references", [])),
                "cross_references":      "|".join(e.get("cross_references", [])),
                "sub_entries_count":     len(e.get("sub_entries", [])),
                "source_page":           e.get("source_page", ""),
                "verified":              e.get("verified", False),
                "needs_review":          e.get("needs_review", False),
            })

    print(f"  ✓ CSV   → {path} ({len(entries)} entries)")
    return path


def save_review_csv(entries: list) -> Path:
    """Save only entries flagged for review — for community verification."""
    flagged = [e for e in entries if e.get("needs_review") or e.get("_enrichment_failed")]
    if not flagged:
        print(f"  ✓ Review CSV → none flagged")
        return None
    return save_csv(flagged, label="needs_review")


def setup_postgres(conn):
    """Create tables if they don't exist."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

            CREATE TABLE IF NOT EXISTS wambule_lexicon (
                id                   UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                headword             TEXT NOT NULL,
                headword_variants    TEXT[],
                phoneme_group        TEXT,
                is_loan_word         BOOLEAN,
                native_or_loan       TEXT,
                part_of_speech       TEXT,
                definition_en        TEXT,
                definition_ne_script TEXT,
                definition_ne_translit TEXT,
                scientific_name      TEXT,
                loan_sources         JSONB DEFAULT '[]',
                lit_references       TEXT[],
                cross_references     TEXT[],
                dialect              TEXT DEFAULT 'Unknown',
                domain_tags          TEXT[],
                sub_entries          JSONB DEFAULT '[]',
                source_page          INTEGER,
                verified             BOOLEAN DEFAULT FALSE,
                needs_review         BOOLEAN DEFAULT FALSE,
                created_at           TIMESTAMP DEFAULT NOW()
            );

            CREATE INDEX IF NOT EXISTS idx_headword     ON wambule_lexicon(headword);
            CREATE INDEX IF NOT EXISTS idx_phoneme      ON wambule_lexicon(phoneme_group);
            CREATE INDEX IF NOT EXISTS idx_pos          ON wambule_lexicon(part_of_speech);
            CREATE INDEX IF NOT EXISTS idx_is_loan      ON wambule_lexicon(is_loan_word);
            CREATE INDEX IF NOT EXISTS idx_verified     ON wambule_lexicon(verified);
            CREATE INDEX IF NOT EXISTS idx_needs_review ON wambule_lexicon(needs_review);
            CREATE INDEX IF NOT EXISTS idx_fts ON wambule_lexicon
                USING GIN (to_tsvector('simple',
                    coalesce(headword,'') || ' ' || coalesce(definition_en,'')));
        """)
        conn.commit()
    print(f"  ✓ PostgreSQL schema ready")


def save_postgres(entries: list, db_config: dict) -> tuple:
    """Insert enriched entries into PostgreSQL."""
    if not POSTGRES_AVAILABLE:
        print(f"  ✗ psycopg2 not installed — skipping PostgreSQL")
        return 0, 0

    conn = psycopg2.connect(**db_config)
    setup_postgres(conn)

    inserted = skipped = 0
    with conn.cursor() as cur:
        for e in entries:
            if e.get("_parse_error") or e.get("_stage1_error"):
                skipped += 1
                continue

            defn = e.get("definition", {})
            ne = defn.get("ne", {}) if isinstance(defn, dict) else {}

            try:
                cur.execute("""
                    INSERT INTO wambule_lexicon (
                        id, headword, headword_variants, phoneme_group,
                        is_loan_word, native_or_loan, part_of_speech,
                        definition_en, definition_ne_script, definition_ne_translit,
                        scientific_name, loan_sources, lit_references,
                        cross_references, dialect, domain_tags, sub_entries,
                        source_page, verified, needs_review
                    ) VALUES (
                        %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s
                    ) ON CONFLICT (id) DO NOTHING
                """, (
                    e.get("id", str(uuid.uuid4())),
                    e.get("headword", e.get("raw_headword", "")),
                    e.get("headword_variants", []),
                    e.get("phoneme_group", ""),
                    e.get("is_loan_word"),
                    e.get("native_or_loan", ""),
                    e.get("part_of_speech", e.get("raw_pos", "")),
                    defn.get("en", "") if isinstance(defn, dict) else str(defn),
                    ne.get("script", "") if isinstance(ne, dict) else "",
                    ne.get("translit", "") if isinstance(ne, dict) else "",
                    e.get("scientific_name"),
                    json.dumps(e.get("loan_sources", []), ensure_ascii=False),
                    e.get("lit_references", []),
                    e.get("cross_references", []),
                    e.get("dialect", "Unknown"),
                    e.get("domain_tags", []),
                    json.dumps(e.get("sub_entries", []), ensure_ascii=False),
                    e.get("source_page"),
                    e.get("verified", False),
                    e.get("needs_review", False),
                ))
                inserted += 1 if cur.rowcount > 0 else 0
                skipped  += 1 if cur.rowcount == 0 else 0
            except Exception as ex:
                skipped += 1

        conn.commit()
    conn.close()

    print(f"  ✓ PostgreSQL → inserted: {inserted}, skipped: {skipped}")
    return inserted, skipped


def save_all(entries: list, db_config: dict = None, label: str = "") -> dict:
    """Save to all outputs. Returns dict of output paths."""
    entries = assign_ids(entries)

    print(f"\n{'─'*60}")
    print(f"  SAVING {len(entries)} ENTRIES...\n")

    paths = {}
    paths["json"] = save_json(entries, label)
    paths["csv"]  = save_csv(entries, label)
    paths["review_csv"] = save_review_csv(entries)

    if db_config:
        save_postgres(entries, db_config)

    return paths