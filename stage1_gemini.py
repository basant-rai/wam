"""
Stage 1: Gemini Extraction
==========================
Sends PDF pages to Gemini API and gets raw JSON with
100% accurate Wambule phoneme characters.
"""

import json
import time
import fitz  # PyMuPDF
from google import genai
from google.genai import types
from PIL import Image
import io

# ── Gemini System Prompt ───────────────────────────────────────────────────────
# This is the "soap" — cleans the characters perfectly
GEMINI_SYSTEM_PROMPT = """You are a linguistic data scientist specializing in 
Tibeto-Burman languages. Extract ALL Wambule lexicon entries from this dictionary 
page into a clean JSON list.

CRITICAL CHARACTER RULES — follow exactly:
- The symbol looking like an inverted V or capital A is strictly ʌ (U+028C)
- Use ONLY these phonemes where they appear: ʌ ā ī ū ɓ ɗ ŋ ɖ ʈ ʔ ś ṣ ñ
- Preserve ALL diacritics: tildes (ã), macrons (ā), dots (ṣ), superscripts (a¹)
- Preserve Devanagari script exactly as printed
- Preserve ALL superscript numbers on headwords (a¹, a², ase¹, ase²)

OUTPUT FORMAT — return ONLY a JSON array, no extra text, no markdown backticks:
[
  {
    "raw_headword": "exact headword including superscripts",
    "raw_variants": ["variant1", "variant2"],
    "raw_pos": "part of speech exactly as printed",
    "raw_etymology": "full etymology bracket content including Devanagari",
    "raw_definition": "full English definition text",
    "raw_nepali": "Nepali equivalent if given after Nep. label",
    "raw_lit_section": "full Lit. reference string if present",
    "raw_cf": "cross-reference targets after Cf. if present",
    "raw_sub_entries": ["any compound forms or sub-entries as strings"],
    "raw_domain_tag": "any domain or dialect tag like [Wamdyal dialect]",
    "raw_scientific": "scientific name if present e.g. Linum usitatissimum"
  }
]

RULES:
- Every entry on the page must be captured — do not skip any
- If a field is absent, use null
- Sub-entries are compound forms like 'ʌngal pacam vt-2a to embrace'
- Do not interpret or fix — preserve raw text accurately
- Return ONLY the JSON array"""


def pdf_page_to_image(pdf_path: str, page_num: int, dpi: int = 200) -> Image.Image:
    """Convert a single PDF page to a PIL Image."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    doc.close()
    return img


def is_page_digital(pdf_path: str, page_num: int) -> bool:
    """Check if page has extractable text (digital) or needs image (scanned)."""
    doc = fitz.open(pdf_path)
    text = doc[page_num].get_text("text").strip()
    doc.close()
    return len(text) > 50


def extract_page_text(pdf_path: str, page_num: int) -> str:
    """Extract raw text from digital PDF page."""
    import pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num]
        width = page.width
        # Handle two-column layout
        left = page.crop((0, 0, width/2, page.height)).extract_text() or ""
        right = page.crop((width/2, 0, width, page.height)
                          ).extract_text() or ""
        return left.strip() + "\n" + right.strip()


def call_gemini_with_image(client, model: str, image: Image.Image, page_num: int) -> list:
    """Send a scanned page image to Gemini and get raw entries."""
    prompt = f"Page {page_num + 1}. {GEMINI_SYSTEM_PROMPT}"
    response = client.models.generate_content(
        model=model,
        contents=[prompt, image]
    )
    return parse_gemini_response(response.text, page_num)


def call_gemini_with_text(client, model: str, text: str, page_num: int) -> list:
    """Send digital page text to Gemini and get raw entries."""
    prompt = f"""Page {page_num + 1} of the Wambule-English lexicon.
Raw extracted text (may have minor formatting artifacts):

{text}

{GEMINI_SYSTEM_PROMPT}"""
    response = client.models.generate_content(model=model, contents=prompt)
    return parse_gemini_response(response.text, page_num)


def parse_gemini_response(response_text: str, page_num: int) -> list:
    """Parse Gemini's response into a list of raw entry dicts."""
    text = response_text.strip()

    # Strip markdown code fences if Gemini added them
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(
            lines[1:-1]) if lines[-1] == "```" else "\n".join(lines[1:])

    try:
        entries = json.loads(text)
        # Tag each entry with source page
        for e in entries:
            e["_source_page"] = page_num + 1
        return entries
    except json.JSONDecodeError as ex:
        print(f"    ⚠ Gemini JSON parse error on page {page_num + 1}: {ex}")
        # Return a placeholder so we know this page needs manual review
        return [{
            "_source_page": page_num + 1,
            "_parse_error": True,
            "_raw_response": response_text[:500]
        }]


def run_gemini_stage(
    pdf_path: str,
    api_key: str,
    start_page: int,
    end_page: int,
    batch_size: int = 5,
    delay_seconds: float = 2.0
) -> list:
    """
    Main Stage 1 function.
    Processes pages start_page to end_page (1-indexed, inclusive).
    Returns list of all raw entries across all pages.
    """
    client = genai.Client(api_key=api_key)
    model = "gemini-2.5-flash"

    all_raw_entries = []
    # Convert to 0-indexed
    pages = list(range(start_page - 1, end_page))

    print(f"\n{'='*60}")
    print(f"  STAGE 1: GEMINI EXTRACTION")
    print(f"  Pages: {start_page} → {end_page} ({len(pages)} pages)")
    print(f"{'='*60}\n")

    for i, page_num in enumerate(pages):
        print(f"  [{i+1:>3}/{len(pages)}] Page {page_num+1}", end=" → ")

        try:
            if is_page_digital(pdf_path, page_num):
                text = extract_page_text(pdf_path, page_num)
                entries = call_gemini_with_text(client, model, text, page_num)
                method = "text"
            else:
                image = pdf_page_to_image(pdf_path, page_num)
                entries = call_gemini_with_image(
                    client, model, image, page_num)
                method = "image"

            # Filter out error placeholders for count display
            valid = [e for e in entries if not e.get("_parse_error")]
            all_raw_entries.extend(entries)
            print(f"[{method.upper():>5}] {len(valid):>3} entries")

        except Exception as ex:
            print(f"[ERROR] {str(ex)[:60]}")
            all_raw_entries.append({
                "_source_page": page_num + 1,
                "_stage1_error": str(ex)
            })

        # Rate limiting — Gemini free tier allows ~2 req/sec
        if i < len(pages) - 1:
            time.sleep(delay_seconds)

    valid_total = len([e for e in all_raw_entries if not e.get(
        "_parse_error") and not e.get("_stage1_error")])
    print(f"\n  Stage 1 complete: {valid_total} raw entries extracted")
    return all_raw_entries
