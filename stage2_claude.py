"""
Stage 2: Claude Enrichment
==========================
Takes Gemini's raw JSON and transforms it into a fully
structured, database-ready schema.
"""

import json
import time
import anthropic

# ── Claude System Prompt ───────────────────────────────────────────────────────
# This is the "comb" — straightens the structure perfectly
CLAUDE_SYSTEM_PROMPT = """You are a linguistic database architect specializing in 
Kiranti languages. You will receive raw lexicon entries extracted from Opgenort's 
Wambule-English Grammar (2004) and transform them into a structured database schema.

CRITICAL: Preserve the ʌ (U+028C) character exactly — never replace with A or Λ.

YOUR TASK — transform each raw entry into this exact schema:
{
  "id": null,
  "headword": "primary headword with superscript if present e.g. ase¹",
  "headword_variants": ["variant1", "variant2"],
  "phoneme_group": "/ʌ/ or /a/ or whichever section this entry belongs to",
  "is_loan_word": true or false,
  "loan_sources": [
    {
      "language": "Nepali or English or Wambule",
      "devanagari": "Devanagari script if present",
      "romanized": "romanized transliteration"
    }
  ],
  "part_of_speech": "full pos string e.g. noun, verb, adv, bound morph",
  "definition": {
    "en": "English definition",
    "ne": {
      "script": "Devanagari Nepali equivalent",
      "translit": "romanized Nepali"
    }
  },
  "scientific_name": "botanical/zoological name or null",
  "lit_references": ["3", "22"],
  "cross_references": ["word1", "word2"],
  "dialect": "Wamdyal or Hilepane or Jhappali or Udayapure or Unknown",
  "domain_tags": ["swamdi speech", "Santa-Bhes terminology"],
  "sub_entries": [
    {
      "compound_form": "ʌngal pacam",
      "part_of_speech": "vt-2a",
      "definition": "to embrace",
      "nepali_equivalent": "अँगालो मार्नु āgālo mārnu"
    }
  ],
  "native_or_loan": "loan or native or calque",
  "source_page": 546,
  "verified": false,
  "needs_review": false
}

RULES:
1. headword_variants — split comma-separated variants from raw_headword into array
2. loan_sources — parse the [<Nep. ...] or [<Wam. ...] bracket into structured objects
3. lit_references — split "3 22 अन्ध्यारो" into ["3","22"], discard the text after numbers
4. cross_references — extract words after "Cf." into array
5. sub_entries — parse raw_sub_entries strings into structured objects with compound_form, pos, definition
6. is_loan_word — true if etymology contains <Nep. or <Eng., false for native Wambule
7. native_or_loan — "loan" for Nepali/English borrowings, "native" for Wambule roots, "calque" for structural borrowings
8. dialect — extract from domain_tag if it says "Wamdyal dialect" or "Hilepane dialect"
9. scientific_name — preserve exactly including genus and species
10. needs_review — set true if any field was ambiguous or unclear

Return ONLY a JSON array of enriched entries. No markdown, no explanation."""


def enrich_batch(client, raw_entries: list, batch_num: int) -> list:
    """Send a batch of raw entries to Claude for enrichment."""
    raw_json = json.dumps(raw_entries, ensure_ascii=False, indent=2)

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=8000,
        system=CLAUDE_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"""Transform these {len(raw_entries)} raw Wambule lexicon entries 
into the structured schema. These are from pages {raw_entries[0].get('_source_page','?')} 
to {raw_entries[-1].get('_source_page','?')} of Opgenort 2004.

Raw entries:
{raw_json}"""
            }
        ]
    )

    response_text = message.content[0].text.strip()

    # Strip markdown fences if present
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        response_text = "\n".join(lines[1:-1]) if lines[-1] == "```" else "\n".join(lines[1:])

    try:
        enriched = json.loads(response_text)
        return enriched
    except json.JSONDecodeError as ex:
        print(f"    ⚠ Claude JSON parse error in batch {batch_num}: {ex}")
        # Return raw entries flagged for manual review
        for e in raw_entries:
            e["_enrichment_failed"] = True
            e["_enrichment_error"] = str(ex)
        return raw_entries


def run_claude_stage(
    raw_entries: list,
    api_key: str,
    entries_per_batch: int = 20,
    delay_seconds: float = 1.0
) -> list:
    """
    Main Stage 2 function.
    Takes all raw entries from Stage 1 and enriches them in batches.
    Returns list of fully structured entries.
    """
    client = anthropic.Anthropic(api_key=api_key)

    # Filter out error entries from stage 1
    valid_entries = [e for e in raw_entries
                     if not e.get("_parse_error") and not e.get("_stage1_error")]
    error_entries = [e for e in raw_entries
                     if e.get("_parse_error") or e.get("_stage1_error")]

    print(f"\n{'='*60}")
    print(f"  STAGE 2: CLAUDE ENRICHMENT")
    print(f"  Entries to enrich: {len(valid_entries)}")
    print(f"  Entries skipped (Stage 1 errors): {len(error_entries)}")
    print(f"{'='*60}\n")

    # Split into batches
    batches = [
        valid_entries[i:i + entries_per_batch]
        for i in range(0, len(valid_entries), entries_per_batch)
    ]

    all_enriched = []

    for i, batch in enumerate(batches):
        pages = f"{batch[0].get('_source_page','?')}-{batch[-1].get('_source_page','?')}"
        print(f"  Batch {i+1:>3}/{len(batches)} (pages {pages}, {len(batch)} entries)", end=" → ")

        try:
            enriched = enrich_batch(client, batch, i + 1)
            all_enriched.extend(enriched)
            print(f"✓ {len(enriched)} enriched")
        except Exception as ex:
            print(f"✗ ERROR: {str(ex)[:50]}")
            for e in batch:
                e["_enrichment_failed"] = True
            all_enriched.extend(batch)

        if i < len(batches) - 1:
            time.sleep(delay_seconds)

    # Add back error entries flagged for manual review
    all_enriched.extend(error_entries)

    needs_review = len([e for e in all_enriched if e.get("needs_review") or e.get("_enrichment_failed")])
    print(f"\n  Stage 2 complete: {len(all_enriched)} total entries")
    print(f"  Flagged for review: {needs_review}")
    return all_enriched