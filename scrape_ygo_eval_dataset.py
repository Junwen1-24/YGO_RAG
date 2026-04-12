#!/usr/bin/env python3
"""
Build a stratified YGO Q&A evaluation CSV from db.ygoresources.com.

Uses the site's documented JSON API (see /about/api); listing pages are JS-driven,
so discovery is via /data/idx/qa/tags. BeautifulSoup is used to normalize any HTML
fragments that may appear in text fields.
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import re
import time
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

BASE = "https://db.ygoresources.com"
TAG_INDEX_URL = f"{BASE}/data/idx/qa/tags"
QA_URL = f"{BASE}/data/qa/{{}}"
CARD_URL = f"{BASE}/data/card/{{}}"

DEFAULT_DELAY = 1.0
OUTPUT_CSV = Path(__file__).resolve().parent / "data" / "ygo_eval_dataset.csv"

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/html;q=0.9, */*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Display names must match user specification
CAT_BASIC = "Basic Information Retrieval"
CAT_PSCT = "PSCT & Logic (Cost vs. Effect)"
CAT_TIMING = 'The "Timing" Challenge (When vs If)'
CAT_CHAINS = "Chains & Spell Speeds"
CAT_COMPLEX = "Complex Rulings & Floodgates"

# Higher tier wins when multiple tags map to different categories ("most complex")
CATEGORY_TIER: dict[str, int] = {
    CAT_BASIC: 1,
    CAT_PSCT: 2,
    CAT_TIMING: 3,
    CAT_CHAINS: 4,
    CAT_COMPLEX: 5,
}

TARGET_COUNTS: dict[str, int] = {
    CAT_CHAINS: 24,
    CAT_TIMING: 18,
    CAT_COMPLEX: 24,
    CAT_PSCT: 24,
    CAT_BASIC: 30,
}

# KONAMI tags (k:...) and org tags (o:...) -> category. Unlisted tags fall back to heuristics / Basic.
TAG_TO_CATEGORY: dict[str, str] = {
    # --- KONAMI (k:) ---
    "k:Basic Rules": CAT_BASIC,
    "k:Normal/Token": CAT_BASIC,
    "k:Monster": CAT_BASIC,
    "k:Flip": CAT_BASIC,
    "k:Toon": CAT_BASIC,
    "k:Union": CAT_BASIC,
    "k:Spirit": CAT_BASIC,
    "k:Gemini": CAT_BASIC,
    "k:Spell": CAT_PSCT,
    "k:Trap": CAT_PSCT,
    "k:Chain": CAT_CHAINS,
    "k:Damage Step": CAT_TIMING,
    "k:Replay": CAT_TIMING,
    "k:Fusion": CAT_COMPLEX,
    "k:Ritual": CAT_COMPLEX,
    "k:Synchro": CAT_COMPLEX,
    "k:Xyz": CAT_COMPLEX,
    "k:Pendulum": CAT_COMPLEX,
    "k:Link": CAT_COMPLEX,
    "k:Special Summon Monster": CAT_COMPLEX,
}

_ORG_COMPLEX = (
    "o:Special Summoning",
    "o:Fusion Summoning",
    "o:Ritual Summoning",
    "o:Synchro Summoning",
    "o:Xyz Summoning",
    "o:Pendulum Summoning",
    "o:Link Summoning",
    "o:Continuous Effect",
    "o:Continuous-ish Spell/Trap Card",
    "o:Continuous current ATK/DEF addition/subtraction",
    "o:Continuous current ATK/DEF setting-to-value",
    "o:Unaffected by Effects",
    "o:When a Monster Would Be Summoned",
    "o:Summons using Material",
    "o:Xyz Material",
    "o:Monster Gains Effect",
    "o:Temporary Banishment",
    "o:Manual Summoning Procedure",
    "o:Tribute Summoning",
    "o:No Available Zones",
    "o:Trap Monster",
    "o:Being Flipped Face-Down",
    "o:Being Currently Face-Down",
    "o:Change of Control",
    "o:Possession",
    "o:Ownership",
)
_ORG_CHAINS = (
    "o:Simultaneous Effects Go On Chain",
    "o:Immediately After Chain Link",
    "o:Negating a Chain Link's Activation",
    "o:Negating a Chain Link's Effect",
    "o:Sending Cards After Chain Resolution",
    "o:Quick Effect",
    "o:Card Activated by Effect",
)
_ORG_TIMING = (
    "o:Damage Step",
    "o:Battle Replay",
    "o:Phase-Trigger Effect",
    "o:Trigger Effect",
    "o:Delayed Effects",
    "o:Skipping a Phase",
    "o:Phase Change by Effect",
    "o:Mandatory Actions during Phase",
)
_ORG_PSCT = (
    "o:Costs",
    "o:Activation Condition",
    "o:Activation Legality",
    "o:Activation Procedure",
    "o:Attempt Legality",
    "o:Optional Part of Effect",
    "o:Mandatory Effect",
    "o:Targeting",
    "o:Negating a card's effects",
    "o:Non-Activated Effect Application",
    "o:Conjunction Success",
    "o:Exact Application Order",
    "o:Tributing",
    "o:Discarding",
    "o:Banishing",
    "o:Sending to the GY",
    "o:Adding Cards to the Hand",
    "o:Returning to the Hand",
    "o:Returning to the Deck",
    "o:Destruction",
    "o:Destruction by Battle",
    "o:Negating Attacks",
    "o:Battle Damage Modification",
    "o:Effect Damage Modification",
    "o:Multiple Attacks",
    "o:Equip Cards",
    "o:Counters",
    "o:Token",
)


def _register_tags(tag_map: dict[str, str], tags: tuple[str, ...], category: str) -> None:
    tier = CATEGORY_TIER[category]
    for t in tags:
        prev = tag_map.get(t)
        if prev is None or tier > CATEGORY_TIER[prev]:
            tag_map[t] = category


_register_tags(TAG_TO_CATEGORY, _ORG_COMPLEX, CAT_COMPLEX)
_register_tags(TAG_TO_CATEGORY, _ORG_CHAINS, CAT_CHAINS)
_register_tags(TAG_TO_CATEGORY, _ORG_TIMING, CAT_TIMING)
_register_tags(TAG_TO_CATEGORY, _ORG_PSCT, CAT_PSCT)

# Remaining org tags from /data/idx/qa/tags (improves categorization when no k: tag maps high)
_ORG_MISC = {
    CAT_COMPLEX: (
        'o:"(...) by a card effect"',
        'o:"(a card) cannot be (...)"',
        'o:"You cannot (...) during the turn (...)"',
        'o:"if (a card) would be (...), you can (...) instead"',
        "o:(Not) Properly Summoned",
        "o:Card Movement Redirection",
        "o:Cards as a Sequence of Copies",
        "o:Changing a Card's Card Type",
        "o:Changing a Card's Properties",
        "o:Different ATK/DEF modifiers interacting",
        "o:Exclusively Self-Applied Modifiers",
        "o:Important Precedent",
        "o:Lingering current ATK/DEF addition/subtraction",
        "o:Lingering current ATK/DEF setting-to-value",
        "o:Lingering original ATK/DEF setting-to-value",
        "o:Multiple Attributes at the same time",
        "o:Non-Standard Victory Condition",
        "o:Pendulum",
        "o:References Printed Properties",
    ),
    CAT_TIMING: ("o:Cannot Conduct the Battle Phase",),
    CAT_BASIC: (
        "o:Effect Monster with no effects",
        "o:Forgetting",
        "o:Gemini Monster",
        "o:Lists a high number of cards",
        "o:Normal Summoning again (Gemini)",
        "o:Reader beware",
        "o:Standard Victory Condition",
        "o:Uncertainty",
    ),
}
for _cat, _tags in _ORG_MISC.items():
    _register_tags(TAG_TO_CATEGORY, _tags, _cat)

CARD_REF_RE = re.compile(r"<<(\d+)>>")
_HTML_LIKE_RE = re.compile(r"<[a-zA-Z][^>]*>")

log = logging.getLogger(__name__)


class ThrottledClient:
    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update(BROWSER_HEADERS)
        self._last_req = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last_req
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)

    def get_json(self, url: str) -> Any:
        self.wait()
        r = self.session.get(url, timeout=90)
        self._last_req = time.monotonic()
        r.raise_for_status()
        return r.json()


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return s.strip()


def strip_html_artifacts(s: str) -> str:
    if not s:
        return ""
    if _HTML_LIKE_RE.search(s):
        return normalize_text(BeautifulSoup(s, "html.parser").get_text(separator=" ", strip=True))
    return normalize_text(s)


def merge_answer_sections(answer: str) -> str:
    parts = [normalize_text(p) for p in answer.split("~~~")]
    parts = [p for p in parts if p]
    return normalize_text("\n\n".join(parts))


def fetch_card_name(client: ThrottledClient, cache: dict[int, str], cid: int) -> str:
    if cid in cache:
        return cache[cid]
    try:
        data = client.get_json(CARD_URL.format(cid))
    except requests.RequestException as e:
        log.debug("Card %s: %s", cid, e)
        cache[cid] = ""
        return ""
    card_data = data.get("cardData") or {}
    en = card_data.get("en") or {}
    name = normalize_text(en.get("name") or "")
    if not name and isinstance(card_data, dict):
        # Fallback to first available locale name if EN is missing.
        for locale_data in card_data.values():
            if not isinstance(locale_data, dict):
                continue
            candidate = normalize_text(locale_data.get("name") or "")
            if candidate:
                name = candidate
                break
    cache[cid] = name
    return name


def resolve_card_refs(text: str, client: ThrottledClient, cache: dict[int, str]) -> str:
    if not text or "<<" not in text:
        return text

    def repl(m: re.Match[str]) -> str:
        cid = int(m.group(1))
        n = fetch_card_name(client, cache, cid)
        return f'"{n}"' if n else f"<<{cid}>>"

    return CARD_REF_RE.sub(repl, text)


def primary_category(tags: list[str]) -> str:
    best_cat = CAT_BASIC
    best_tier = 0
    for t in tags:
        c = TAG_TO_CATEGORY.get(t)
        if not c:
            continue
        tier = CATEGORY_TIER[c]
        if tier > best_tier:
            best_tier = tier
            best_cat = c
    if best_tier > 0:
        return best_cat
    joined = " ".join(tags).lower()
    blob = joined
    if any(x in blob for x in ("damage step", "replay", "when ", " if ", "trigger")):
        return CAT_TIMING
    if "chain" in blob or "spell speed" in blob:
        return CAT_CHAINS
    if any(x in blob for x in ("cost", "activate", "activation", "effect")):
        return CAT_PSCT
    return CAT_BASIC


def konami_tag_labels(tags: list[str]) -> list[str]:
    return sorted({t[2:].strip() for t in tags if t.startswith("k:")})


def is_informative_answer(answer: str, min_len: int = 40) -> bool:
    a = normalize_text(answer)
    if len(a) < min_len:
        return False
    trivial = re.sub(r"[^a-z]", "", a.lower())
    if trivial in {"yes", "no", "yesno"}:
        return False
    return True


def question_fingerprint(q: str) -> str:
    q = q.lower()
    q = re.sub(r"[^a-z0-9\s]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def near_duplicate(a: str, b: str, threshold: float = 0.88) -> bool:
    if not a or not b:
        return False
    return SequenceMatcher(None, a, b).ratio() >= threshold


def total_target_count(targets: dict[str, int]) -> int:
    return sum(targets.values())


def parse_qa_entry(
    client: ThrottledClient,
    card_cache: dict[int, str],
    raw: dict[str, Any],
) -> dict[str, Any] | None:
    qa_data = raw.get("qaData") or {}
    en = qa_data.get("en")
    if not isinstance(en, dict):
        return None

    status = en.get("translationStatus")
    if status != "confirmed":
        return None

    question = merge_answer_sections(en.get("question") or "")
    answer = merge_answer_sections(en.get("answer") or "")
    question = strip_html_artifacts(question)
    answer = strip_html_artifacts(answer)

    if not question or not answer:
        return None
    if not is_informative_answer(answer):
        return None

    tags = raw.get("tags")
    if not isinstance(tags, list):
        tags = []
    tags_s = [t for t in tags if isinstance(t, str)]

    cards = raw.get("cards")
    card_name = ""
    if isinstance(cards, list) and cards:
        first = cards[0]
        if isinstance(first, int):
            card_name = fetch_card_name(client, card_cache, first)
    if not card_name:
        m = CARD_REF_RE.search(en.get("question") or "")
        if m:
            card_name = fetch_card_name(client, card_cache, int(m.group(1)))

    cat = primary_category(tags_s)
    k_labels = konami_tag_labels(tags_s)

    return {
        "ID": str(en.get("id", "")),
        "Card_Name": card_name,
        "Question": question,
        "Official_Answer": answer,
        "Category": cat,
        "Tags": "|".join(k_labels),
        "_tags_full": tags_s,
        "_fp": question_fingerprint(question),
    }


def stratified_select(
    pools: dict[str, list[dict[str, Any]]],
    targets: dict[str, int],
    rng: random.Random,
    dup_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, int], list[str]]:
    warnings: list[str] = []
    # Scarce categories first to reduce contention on dedupe
    order = sorted(targets.keys(), key=lambda c: targets[c])
    selected: list[dict[str, Any]] = []
    global_fps: list[str] = []
    card_counts: defaultdict[str, int] = defaultdict(int)

    def sort_key(row: dict[str, Any]) -> tuple[int, int, float]:
        cn = row.get("Card_Name") or ""
        return (card_counts[cn], -len(row.get("_tags_full") or []), row.get("_tie", 0.0))

    per_cat_counts: dict[str, int] = {c: 0 for c in targets}

    for cat in order:
        need = targets[cat]
        pool = list(pools.get(cat) or [])
        for row in pool:
            row["_tie"] = rng.random()
        unpicked = list(pool)
        taken = 0
        while taken < need and unpicked:
            unpicked.sort(key=sort_key)
            row = unpicked.pop(0)
            fp = row["_fp"]
            if any(near_duplicate(fp, g, dup_threshold) for g in global_fps):
                continue
            selected.append(row)
            global_fps.append(fp)
            cn = row.get("Card_Name") or ""
            card_counts[cn] += 1
            taken += 1
        per_cat_counts[cat] = taken
        if taken < need:
            warnings.append(
                f"Category {cat!r}: only collected {taken} of {need} required "
                f"(pool size {len(pool)})."
            )

    return selected, per_cat_counts, warnings


def fill_shortfalls(
    selected: list[dict[str, Any]],
    per_cat_counts: dict[str, int],
    pools: dict[str, list[dict[str, Any]]],
    targets: dict[str, int],
    rng: random.Random,
    dup_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, int], list[str]]:
    warnings: list[str] = []
    by_id = {r["ID"] for r in selected}
    fps = [r["_fp"] for r in selected]
    card_counts: defaultdict[str, int] = defaultdict(int)
    for row in selected:
        card_counts[row.get("Card_Name") or ""] += 1

    def sort_key(row: dict[str, Any]) -> tuple[int, int, float]:
        return (
            card_counts[row.get("Card_Name") or ""],
            -len(row.get("_tags_full") or []),
            row.get("_tie", 0.0),
        )

    # Fill by category first.
    for cat in sorted(targets.keys(), key=lambda c: targets[c] - per_cat_counts.get(c, 0), reverse=True):
        need = max(0, targets[cat] - per_cat_counts.get(cat, 0))
        if need <= 0:
            continue
        pool = [r for r in pools.get(cat, []) if r["ID"] not in by_id]
        for row in pool:
            row["_tie"] = rng.random()

        while need > 0 and pool:
            pool.sort(key=sort_key)
            row = pool.pop(0)
            if any(near_duplicate(row["_fp"], fp, dup_threshold) for fp in fps):
                continue
            selected.append(row)
            by_id.add(row["ID"])
            fps.append(row["_fp"])
            per_cat_counts[cat] = per_cat_counts.get(cat, 0) + 1
            card_counts[row.get("Card_Name") or ""] += 1
            need -= 1

    # Then top-up globally (if still below total target) with relaxed constraints.
    need_total = total_target_count(targets) - len(selected)
    if need_total > 0:
        all_remaining = [
            r for cat_pool in pools.values() for r in cat_pool if r["ID"] not in by_id
        ]
        for row in all_remaining:
            row["_tie"] = rng.random()
        all_remaining.sort(key=sort_key)
        for row in all_remaining:
            if need_total <= 0:
                break
            if any(near_duplicate(row["_fp"], fp, dup_threshold) for fp in fps):
                continue
            selected.append(row)
            by_id.add(row["ID"])
            fps.append(row["_fp"])
            cat = row["Category"]
            per_cat_counts[cat] = per_cat_counts.get(cat, 0) + 1
            card_counts[row.get("Card_Name") or ""] += 1
            need_total -= 1

    # Final diagnostics for strict target checking.
    for cat, target in targets.items():
        have = per_cat_counts.get(cat, 0)
        if have < target:
            warnings.append(f"Category {cat!r}: only {have}/{target} after refill attempts.")
    if len(selected) < total_target_count(targets):
        warnings.append(
            f"Total rows only {len(selected)}/{total_target_count(targets)} after refill attempts."
        )
    return selected, per_cat_counts, warnings


def main() -> int:
    ap = argparse.ArgumentParser(description="Scrape YGO Q&A eval dataset from db.ygoresources.com")
    ap.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help=f"Seconds between HTTP requests (default {DEFAULT_DELAY})",
    )
    ap.add_argument(
        "--max-fetches",
        type=int,
        default=12000,
        help="Max Q&A JSON fetches (stops early when pools are full; ~10k+ typical for 120 rows)",
    )
    ap.add_argument(
        "--pool-factor",
        type=int,
        default=12,
        help="Stop growing per-category pool after target * this many candidates",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for shuffling and tie-breaking",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_CSV,
        help="Output CSV path",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Log progress every N Q&A fetches",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    rng = random.Random(args.seed)
    client = ThrottledClient(args.delay)
    card_cache: dict[int, str] = {}

    log.info("Fetching Q&A tag index…")
    try:
        tag_index = client.get_json(TAG_INDEX_URL)
    except requests.RequestException as e:
        log.error("Failed to load tag index: %s", e)
        return 1

    all_ids = sorted({i for v in tag_index.values() for i in v if isinstance(i, int)})
    rng.shuffle(all_ids)
    log.info("Unique Q&A IDs in index: %d", len(all_ids))

    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pool_limits = {c: n * args.pool_factor for c, n in TARGET_COUNTS.items()}
    fetches = 0
    skipped_fetch = 0

    for qid in all_ids:
        if fetches >= args.max_fetches:
            log.warning("Reached --max-fetches=%d before filling all pools.", args.max_fetches)
            break
        if all(len(pools[c]) >= pool_limits[c] for c in TARGET_COUNTS):
            log.info("Pool targets satisfied; stopping fetch phase.")
            break

        try:
            raw = client.get_json(QA_URL.format(qid))
        except requests.RequestException as e:
            log.debug("QA %s fetch error: %s", qid, e)
            fetches += 1
            continue

        fetches += 1
        if args.progress_every > 0 and (fetches % args.progress_every == 0):
            log.info(
                "Progress: fetched %d | pools=%s",
                fetches,
                {c: len(pools[c]) for c in TARGET_COUNTS},
            )
        parsed = parse_qa_entry(client, card_cache, raw)
        if not parsed:
            continue

        cat = parsed["Category"]
        if len(pools[cat]) < pool_limits[cat]:
            pools[cat].append(parsed)
        else:
            skipped_fetch += 1

    log.info("Q&A fetches: %d; pool-saturated skips (add): %d", fetches, skipped_fetch)

    selected, per_cat, warns = stratified_select(pools, TARGET_COUNTS, rng, dup_threshold=0.88)
    target_total = total_target_count(TARGET_COUNTS)
    if len(selected) < target_total:
        log.info(
            "Initial selection produced %d/%d rows. Running refill pass…",
            len(selected),
            target_total,
        )
        selected, per_cat, refill_warns = fill_shortfalls(
            selected=selected,
            per_cat_counts=per_cat,
            pools=pools,
            targets=TARGET_COUNTS,
            rng=rng,
            dup_threshold=0.95,
        )
        warns.extend(refill_warns)

    log.info("Resolving card placeholders in %d selected rows…", len(selected))
    for row in selected:
        row["Question"] = resolve_card_refs(row["Question"], client, card_cache)
        row["Official_Answer"] = resolve_card_refs(row["Official_Answer"], client, card_cache)
        if not row.get("Card_Name"):
            m = CARD_REF_RE.search(row["Question"])
            if m:
                row["Card_Name"] = fetch_card_name(client, card_cache, int(m.group(1)))

    for w in warns:
        log.warning("%s", w)

    log.info("Per-category selected counts:")
    for c in TARGET_COUNTS:
        log.info("  %s: %d (target %d)", c, per_cat.get(c, 0), TARGET_COUNTS[c])

    fieldnames = ["ID", "Card_Name", "Question", "Official_Answer", "Category", "Tags"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in selected:
            w.writerow({k: row[k] for k in fieldnames})

    unresolved_q = sum(1 for r in selected if "<<" in r["Question"])
    unresolved_a = sum(1 for r in selected if "<<" in r["Official_Answer"])
    if unresolved_q or unresolved_a:
        log.warning(
            "Unresolved card placeholders remain: question=%d answer=%d",
            unresolved_q,
            unresolved_a,
        )

    log.info("Wrote %d rows to %s", len(selected), args.output)
    if len(selected) != target_total:
        log.error(
            "Dataset size mismatch: expected %d rows, got %d rows.",
            target_total,
            len(selected),
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
