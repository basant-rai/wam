"""
Wambule Language Data Extraction Pipeline
==========================================
Handles mixed PDFs (digital + scanned pages)
Outputs to: JSON, CSV, and PostgreSQL

Usage:
    python extractor.py --pdf path/to/opgenort.pdf
    python extractor.py --pdf path/to/opgenort.pdf --pages 700-900
    python extractor.py --pdf path/to/opgenort.pdf --no-db  (skip PostgreSQL)
"""

import argparse
import csv
import json
import os
import re
import uuid
from pathlib import Path
from datetime import datetime

import pdfplumber
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

# ─── Optional PostgreSQL ───────────────────────────────────────────────────────
try:
    import psycopg2
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Tesseract config: Nepali + English for mixed Devanagari/Latin pages
TESS_CONFIG = "--oem 3 --psm 6"
TESS_LANG = "nep+eng"

# PostgreSQL connection (update these or use environment variables)
DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     os.getenv("DB_PORT", "5432"),
    "dbname":   os.getenv("DB_NAME", "wambule"),
    "user":     os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASS", ""),
}

# Wambule parts of speech abbreviations (from Opgenort)
POS_MAP = {
    "n":    "noun",
    "v":    "verb",
    "adj":  "adjective",
    "adv":  "adverb",
    "pro":  "pronoun",
    "pp":   "postposition",
    "conj": "conjunction",
    "num":  "numeral",
    "prt":  "particle",
    "dem":  "demonstrative",
    "int":  "interjection",
    "aux":  "auxiliary",
}

# Known dialects
DIALECTS = ["Wamdyal", "Hilepane", "Jhappali", "Udayapure", "Unknown"]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: PAGE CLASSIFIER
# Determines if a page is digital text or a scanned image
# ══════════════════════════════════════════════════════════════════════════════

def classify_page(page) -> str:
    """
    Returns 'digital' if the page has extractable text,
    'scanned' if it's an image that needs OCR.
    """
    text = page.get_text("text").strip()
    # If less than 20 characters, it's likely a scanned image
    if len(text) < 20:
        return "scanned"
    return "digital"


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: TEXT EXTRACTION
# Two modes: direct text (digital) or OCR (scanned)
# ══════════════════════════════════════════════════════════════════════════════

def extract_text_digital(pdf_path: str, page_num: int) -> str:
    """Extract text directly from digital PDF pages using pdfplumber."""
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num]

        # Multi-column detection: split page into left and right halves
        width = page.width
        left_bbox = (0, 0, width / 2, page.height)
        right_bbox = (width / 2, 0, width, page.height)

        left_text = page.crop(left_bbox).extract_text() or ""
        right_text = page.crop(right_bbox).extract_text() or ""

        # Combine column text naturally
        combined = left_text.strip() + "\n" + right_text.strip()
        return combined


def extract_text_ocr(pdf_path: str, page_num: int) -> str:
    """Run Tesseract OCR on a scanned page."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]

    # Render at 300 DPI for good OCR accuracy
    mat = fitz.Matrix(300 / 72, 300 / 72)
    pix = page.get_pixmap(matrix=mat)
    img = Image.open(io.BytesIO(pix.tobytes("png")))

    text = pytesseract.image_to_string(img, lang=TESS_LANG, config=TESS_CONFIG)
    doc.close()
    return text


def extract_page_text(pdf_path: str, page_num: int) -> tuple[str, str]:
    """
    Master extractor: auto-detects page type and uses the right method.
    Returns (text, method_used)
    """
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    page_type = classify_page(page)
    doc.close()

    if page_type == "digital":
        text = extract_text_digital(pdf_path, page_num)
        print(text)

        return text, "digital"
    else:
        text = extract_text_ocr(pdf_path, page_num)
        return text, "ocr"


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: LEXICON PARSER
# Parses raw text into structured Wambule entries
# ══════════════════════════════════════════════════════════════════════════════

def normalize_word(word: str) -> str:
    """
    Create a search-friendly version of the word by removing diacritics.
    e.g., 'kāi' → 'kai', 'ā:' → 'a:'
    """
    replacements = {
        "ā": "a", "ī": "i", "ū": "u",
        "ɓ": "b", "ɗ": "d", "ŋ": "ng",
        "ɖ": "d", "ʈ": "t", "ʔ": "'",
    }
    result = word
    for char, replacement in replacements.items():
        result = result.replace(char, replacement)
    return result.lower().strip()


def detect_dialect(text: str) -> str:
    """Detect dialect markers in text if present."""
    text_lower = text.lower()
    for dialect in DIALECTS:
        if dialect.lower() in text_lower:
            return dialect
    return "Unknown"


def parse_lexicon_entry(line: str, source: str = "Opgenort 2004") -> dict | None:
    """
    Parses a single lexicon line into a structured entry.

    Opgenort's lexicon typically follows this pattern:
        word [pos] definition (Nepali equivalent)

    Example:
        kaai n. water (पानी)

    Returns None if the line doesn't look like a lexicon entry.
    """
    line = line.strip()
    if not line or len(line) < 3:
        return None

    # Pattern: word + optional POS + definition
    # Matches: "kaai n. water" or "kaai water" or "kaai (n) water"
    pattern = r"""
        ^(?P<word>[a-zA-Zāīūɓɗŋɖʈʔ\u0900-\u097F'\-]+)   # Wambule word
        \s*
        (?:[\(\[]?(?P<pos>[a-z]{1,5})[\.\)\]]?\s+)?        # Optional POS tag
        (?P<definition>.+)$                                  # Definition
    """
    match = re.match(pattern, line, re.VERBOSE | re.UNICODE)
    if not match:
        return None

    word = match.group("word").strip()
    pos_raw = (match.group("pos") or "").lower().strip(".,()[]")
    definition = match.group("definition").strip()

    # Skip lines that are clearly not lexicon entries
    if len(word) < 2 or len(definition) < 2:
        return None
    if word.lower() in ["the", "and", "or", "of", "in", "see"]:
        return None

    # Extract Nepali meaning if present in parentheses or after a slash
    nepali_match = re.search(
        r'[\(\[（]([^\)\]）]+[\u0900-\u097F][^\)\]）]*)[\)\]）]', definition)
    nepali_meaning = nepali_match.group(1).strip() if nepali_match else ""

    # Clean English definition
    english_def = re.sub(r'[\(\[（][^\)\]）]*[\)\]）]', '', definition).strip()
    english_def = re.sub(r'\s+', ' ', english_def).strip(".,; ")

    # Map POS abbreviation to full form
    pos_full = POS_MAP.get(pos_raw, pos_raw if pos_raw else "unknown")

    return {
        "id":             str(uuid.uuid4()),
        "word":           word,
        "normalized":     normalize_word(word),
        "part_of_speech": pos_full,
        "definitions": [
            {"language": "en", "text": english_def},
            {"language": "ne", "text": nepali_meaning},
        ],
        "dialect":        "Unknown",
        "examples":       [],
        "source":         source,
        "verified":       False,
        "needs_review":   True,
        "extraction_method": "",  # filled in by caller
    }


def parse_page_text(text: str, page_num: int, method: str) -> list[dict]:
    """Parse all lexicon entries from a page's extracted text."""
    entries = []
    lines = text.split("\n")

    for line in lines:
        entry = parse_lexicon_entry(
            line, source=f"Opgenort 2004, p.{page_num + 1}")
        if entry:
            entry["extraction_method"] = method
            entries.append(entry)

    return entries


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: STORAGE LAYER
# Save to JSON, CSV, and PostgreSQL
# ══════════════════════════════════════════════════════════════════════════════

def save_json(entries: list[dict], filename: str = None):
    """Save all entries to a JSON file."""
    path = OUTPUT_DIR / (filename or f"wambule_lexicon_{TIMESTAMP}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"  ✓ JSON saved → {path} ({len(entries)} entries)")
    return path


def save_csv(entries: list[dict], filename: str = None):
    """Save all entries to a CSV file (flattened for spreadsheet use)."""
    path = OUTPUT_DIR / (filename or f"wambule_lexicon_{TIMESTAMP}.csv")

    fieldnames = [
        "id", "word", "normalized", "part_of_speech",
        "english_definition", "nepali_definition",
        "dialect", "source", "verified", "needs_review", "extraction_method"
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for entry in entries:
            en_def = next(
                (d["text"] for d in entry["definitions"] if d["language"] == "en"), "")
            ne_def = next(
                (d["text"] for d in entry["definitions"] if d["language"] == "ne"), "")

            writer.writerow({
                "id":                  entry["id"],
                "word":                entry["word"],
                "normalized":          entry["normalized"],
                "part_of_speech":      entry["part_of_speech"],
                "english_definition":  en_def,
                "nepali_definition":   ne_def,
                "dialect":             entry["dialect"],
                "source":              entry["source"],
                "verified":            entry["verified"],
                "needs_review":        entry["needs_review"],
                "extraction_method":   entry["extraction_method"],
            })

    print(f"  ✓ CSV saved → {path} ({len(entries)} entries)")
    return path


def setup_postgres(conn):
    """Create the wambule_lexicon table if it doesn't exist."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS wambule_lexicon (
                id                  UUID PRIMARY KEY,
                word                TEXT NOT NULL,
                normalized          TEXT,
                part_of_speech      TEXT,
                english_definition  TEXT,
                nepali_definition   TEXT,
                dialect             TEXT DEFAULT 'Unknown',
                examples            JSONB DEFAULT '[]',
                source              TEXT,
                verified            BOOLEAN DEFAULT FALSE,
                needs_review        BOOLEAN DEFAULT TRUE,
                extraction_method   TEXT,
                created_at          TIMESTAMP DEFAULT NOW(),
                updated_at          TIMESTAMP DEFAULT NOW()
            );

            CREATE INDEX IF NOT EXISTS idx_wambule_word       ON wambule_lexicon(word);
            CREATE INDEX IF NOT EXISTS idx_wambule_normalized ON wambule_lexicon(normalized);
            CREATE INDEX IF NOT EXISTS idx_wambule_pos        ON wambule_lexicon(part_of_speech);
            CREATE INDEX IF NOT EXISTS idx_wambule_verified   ON wambule_lexicon(verified);
        """)
        conn.commit()
    print("  ✓ PostgreSQL table ready")


def save_postgres(entries: list[dict], conn):
    """Insert entries into PostgreSQL, skipping duplicates."""
    inserted = 0
    skipped = 0

    with conn.cursor() as cur:
        for entry in entries:
            en_def = next(
                (d["text"] for d in entry["definitions"] if d["language"] == "en"), "")
            ne_def = next(
                (d["text"] for d in entry["definitions"] if d["language"] == "ne"), "")

            try:
                cur.execute("""
                    INSERT INTO wambule_lexicon
                        (id, word, normalized, part_of_speech,
                         english_definition, nepali_definition,
                         dialect, examples, source,
                         verified, needs_review, extraction_method)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (id) DO NOTHING
                """, (
                    entry["id"], entry["word"], entry["normalized"],
                    entry["part_of_speech"], en_def, ne_def,
                    entry["dialect"], json.dumps(entry["examples"]),
                    entry["source"], entry["verified"],
                    entry["needs_review"], entry["extraction_method"]
                ))
                if cur.rowcount > 0:
                    inserted += 1
                else:
                    skipped += 1
            except Exception as e:
                skipped += 1

        conn.commit()

    print(f"  ✓ PostgreSQL → inserted: {inserted}, skipped: {skipped}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(pdf_path: str, page_range: tuple = None, use_db: bool = False):
    """
    Master pipeline:
    1. Open PDF
    2. For each page: classify → extract → parse → collect
    3. Save to all outputs
    """
    print(f"\n{'='*60}")
    print(f"  WAMBULE EXTRACTION PIPELINE")
    print(f"  PDF: {pdf_path}")
    print(f"{'='*60}\n")

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    doc.close()

    # Determine which pages to process
    if page_range:
        start, end = page_range
        pages = range(max(0, start - 1), min(end, total_pages))
    else:
        pages = range(total_pages)

    print(f"Processing {len(pages)} pages out of {total_pages} total...\n")

    all_entries = []
    stats = {"digital": 0, "ocr": 0, "entries": 0, "errors": 0}

    for i, page_num in enumerate(pages):
        try:
            print(f"  Page {page_num + 1:>4} / {total_pages}", end=" → ")
            text, method = extract_page_text(pdf_path, page_num)
            entries = parse_page_text(text, page_num, method)

            stats[method] += 1
            stats["entries"] += len(entries)
            all_entries.extend(entries)

            print(f"[{method.upper():>7}] {len(entries):>3} entries found")

        except Exception as e:
            stats["errors"] += 1
            print(f"[ERROR] {str(e)[:50]}")

    # ── Save to all outputs ──────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  SAVING {len(all_entries)} ENTRIES...\n")

    save_json(all_entries)
    save_csv(all_entries)

    if use_db and POSTGRES_AVAILABLE:
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            setup_postgres(conn)
            save_postgres(all_entries, conn)
            conn.close()
        except Exception as e:
            print(f"  ✗ PostgreSQL error: {e}")
            print(f"    (Run without --db or check your DB_CONFIG settings)")
    elif use_db:
        print("  ✗ psycopg2 not installed — skipping PostgreSQL")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  EXTRACTION COMPLETE")
    print(f"  Digital pages : {stats['digital']}")
    print(f"  OCR pages     : {stats['ocr']}")
    print(f"  Total entries : {stats['entries']}")
    print(f"  Errors        : {stats['errors']}")
    print(f"  Output dir    : {OUTPUT_DIR.resolve()}")
    print(f"{'='*60}\n")

    return all_entries


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_page_range(s: str) -> tuple:
    """Parse '700-900' into (700, 900)."""
    if "-" in s:
        parts = s.split("-")
        return (int(parts[0]), int(parts[1]))
    return (int(s), int(s))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Wambule Language PDF Extraction Pipeline"
    )
    parser.add_argument("--pdf",   required=True, help="Path to the PDF file")
    parser.add_argument("--pages", default=None,
                        help="Page range e.g. 700-900")
    parser.add_argument("--db",    action="store_true",
                        help="Save to PostgreSQL")

    args = parser.parse_args()

    page_range = parse_page_range(args.pages) if args.pages else None
    run_pipeline(args.pdf, page_range=page_range, use_db=args.db)
