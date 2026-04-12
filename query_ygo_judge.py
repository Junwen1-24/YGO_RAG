from __future__ import annotations

import argparse
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


SYSTEM_PROMPT = (
    "You are a professional Yu-Gi-Oh! judge. You must reason only from the retrieved card data "
    "and rule passages provided in the user message. If snippets are insufficient, explicitly say "
    "what evidence is missing instead of guessing. If there is any conflict, prioritize the "
    "comprehensive rulebook passages in context."
)

SYSTEM_PROMPT_NO_RAG = (
    "You are a professional Yu-Gi-Oh! judge. Answer from established Yu-Gi-Oh! TCG rules "
    "and card text conventions. Explain your reasoning clearly."
)

COVERAGE_SYSTEM_PROMPT = (
    "You audit whether retrieved snippets can answer a Yu-Gi-Oh! TCG ruling question. "
    "Use ONLY the provided CARD and RULE snippets—no outside knowledge. "
    "Decide if they contain enough to determine a clear Yes/No or Can/Cannot for the scenario asked.\n"
    "Reply with a single JSON object only, no markdown:\n"
    '{"sufficient": true or false, "reason": "one short sentence", '
    '"missing_evidence": "empty string if sufficient; otherwise name what is missing"}'
)

RULE_TOPICS = {
    "damage step": ["damage", "step", "calculation", "battle", "during"],
    "chain": ["chain", "link", "response", "resolve", "activation"],
    "summon": ["summon", "normal", "special", "flip", "tribute", "set"],
    "activation": ["activate", "activation", "activated", "negate", "cost", "target"],
    "timing": ["when", "if", "miss", "timing", "after", "then", "simultaneous"],
}

# Map high-level tags/keywords to rulebook paths we want to prioritize.
TAG_FORCED_RULE_PATH_HINTS: dict[str, list[str]] = {
    "damage step": [
        "build_site/sections/08_duel_and_turn_structure/05_battle_phase/03_damage_step.html",
    ],
    "chain": [
        "build_site/sections/07_general_game_mechanics/chains.html",
    ],
    "activation": [
        "build_site/sections/11_activation_condition/",
    ],
}
STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "to",
    "and",
    "or",
    "in",
    "on",
    "for",
    "with",
    "is",
    "are",
    "can",
    "this",
    "that",
    "be",
    "it",
    "as",
    "by",
    "from",
    "at",
    "if",
    "then",
}


def ensure_openai_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if api_key:
        return api_key
    raise EnvironmentError(
        "OPENAI_API_KEY is missing. Set it in the process environment, for example: "
        "export OPENAI_API_KEY='sk-...' (do not commit keys to the repository)."
    )


@lru_cache(maxsize=2)
def _load_faiss(index_dir: str, embedding_model: str) -> FAISS:
    path = Path(index_dir)
    if not path.exists():
        raise FileNotFoundError(f"FAISS index directory not found: {path}")

    embeddings = OpenAIEmbeddings(model=embedding_model)
    return FAISS.load_local(
        folder_path=str(path),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )


def _get_card_name(doc: Document) -> str:
    meta_name = doc.metadata.get("card_name")
    if isinstance(meta_name, str) and meta_name.strip():
        return meta_name.strip()

    for line in doc.page_content.splitlines():
        if line.lower().startswith("name:"):
            return line.split(":", 1)[1].strip() or "Unknown Card"

    return "Unknown Card"


def _get_rule_ref(doc: Document) -> str:
    rule_number = doc.metadata.get("rule_number")
    if isinstance(rule_number, str) and rule_number.strip():
        return rule_number.strip()

    section_title = doc.metadata.get("section_title")
    source = doc.metadata.get("source", "unknown_source")
    if isinstance(section_title, str) and section_title.strip():
        return f"{section_title} ({source})"
    return str(source)


def _get_qa_ref(doc: Document) -> str:
    qid = str(doc.metadata.get("qa_id", "")).strip()
    question = str(doc.metadata.get("question", "")).strip()
    if qid:
        return f"QA#{qid}: {question[:80]}"
    return f"QA: {question[:80]}"


def _format_context(docs: list[Document], prefix: str) -> str:
    blocks: list[str] = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown_source")
        blocks.append(f"[{prefix} {i}] source={source}\n{doc.page_content}")
    return "\n\n".join(blocks)


def _normalize_tags(tags: str | list[str] | None) -> list[str]:
    if tags is None:
        return []
    if isinstance(tags, str):
        raw = [part.strip() for part in tags.split("|")]
        return [t for t in raw if t]
    if isinstance(tags, list):
        return [str(t).strip() for t in tags if str(t).strip()]
    return []


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t and t not in STOPWORDS]


def _extract_quoted_terms(text: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for m in re.findall(r'"([^"]{2,80})"', text):
        q = m.strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    return out


def _infer_rule_topics(question: str, tags_list: list[str]) -> list[str]:
    blob = f"{question} {' '.join(tags_list)}".lower()
    found: list[str] = []
    for topic in RULE_TOPICS:
        if topic in blob and topic not in found:
            found.append(topic)
    return found


def _build_retrieval_queries(question: str, tags_list: list[str]) -> tuple[list[str], list[str], list[str]]:
    topics = _infer_rule_topics(question, tags_list)
    card_names = _extract_quoted_terms(question)

    hints: list[str] = []
    hints.extend(topics)
    hints.extend(card_names[:4])
    for t in _tokenize(question):
        if t in {"damage", "step", "chain", "summon", "activate", "negate", "cost", "target"}:
            hints.append(t)

    short_q = " ".join(dict.fromkeys(hints)).strip()
    queries = [question]
    if short_q and short_q.lower() != question.lower():
        queries.append(short_q)
    return queries, topics, card_names


def _build_rule_queries(question: str, tags_list: list[str], base_queries: list[str]) -> list[str]:
    """Tag-aware extra queries so rule retrieval matches how judges search by topic."""
    out: list[str] = []
    seen: set[str] = set()
    for q in base_queries:
        q = q.strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    if tags_list:
        tag_join = " ".join(t.strip() for t in tags_list if t.strip())
        if tag_join:
            combo = f"{question} {tag_join}".strip()
            if combo.lower() not in seen:
                seen.add(combo.lower())
                out.append(combo)
        for t in tags_list[:6]:
            t = t.strip()
            if not t:
                continue
            sq = f"{question} {t}".strip()
            if sq.lower() not in seen:
                seen.add(sq.lower())
                out.append(sq)
            # Topic-only retrieval (helps when the question is long/noisy)
            if t.lower() not in seen:
                seen.add(t.lower())
                out.append(t)
    return out[:10]


def _rule_source_quality_multiplier(doc: Document) -> float:
    """
    Prefer structured rulebook HTML under build_site/sections; de-prioritize shallow index pages.
    Returns multiplier in (0, 1].
    """
    src = str(doc.metadata.get("source", "")).lower().replace("\\", "/")
    if "build_site/sections" in src:
        return 1.0
    if "ygoresources_rulings_list" in src or "/rulings" in src:
        return 0.82
    if "glossary" in src and "build_site" in src:
        return 0.88
    # Root or generic index / landing HTML (navigation-heavy, weak for precise rulings)
    if src.endswith("/index.html") or src.endswith("/index.htm"):
        return 0.14
    if src.endswith("index.html") or src.endswith("index.htm"):
        return 0.18
    if "/index.html" in src:
        return 0.22
    return 0.72


def _tag_rule_match_score(doc: Document, tags_list: list[str]) -> float:
    """Boost chunks whose path, heading, or body match dataset Tags (e.g. Damage Step)."""
    if not tags_list:
        return 0.0
    title = str(doc.metadata.get("section_title", ""))
    src = str(doc.metadata.get("source", ""))
    body = doc.page_content[:6000]
    blob = f"{src} {title} {body}".lower()
    total = 0.0
    for tag in tags_list:
        raw = tag.strip()
        if not raw:
            continue
        low = raw.lower()
        if low in blob:
            total += 1.0
            continue
        parts = [p for p in re.split(r"[^\w]+", low) if len(p) >= 3]
        if not parts:
            continue
        hits = sum(1 for p in parts if p in blob)
        total += 0.55 * (hits / len(parts))
    return total / max(len(tags_list), 1)


def _tag_forced_path_boost(doc: Document, tags_list: list[str]) -> float:
    """
    Extra boost when a rule chunk lives under a path we know is strongly relevant
    for the given Tags (e.g. Damage Step or Chains pages).
    """
    if not tags_list:
        return 0.0
    src = str(doc.metadata.get("source", "")).lower().replace("\\", "/")
    if not src:
        return 0.0
    score = 0.0
    low_tags = [t.lower() for t in tags_list]
    for raw_tag in low_tags:
        for topic, paths in TAG_FORCED_RULE_PATH_HINTS.items():
            if topic not in raw_tag:
                continue
            for p in paths:
                if p.lower() in src:
                    # Strong boost if we hit an explicitly mapped path.
                    score += 1.0
    return min(score, 2.0)


def _retrieve_rule_candidates(
    *,
    store: FAISS,
    question: str,
    base_queries: list[str],
    tags_list: list[str],
    final_k: int,
    fetch_k: int,
    keyword_k: int,
    keyword_terms: list[str],
) -> list[Document]:
    """
    Hybrid retrieval for rules, then rerank with tag match + shallow-HTML penalty.
    """
    rule_queries = _build_rule_queries(question, tags_list, base_queries)
    tag_tokens: list[str] = []
    for t in tags_list:
        tag_tokens.extend(_tokenize(t))
    merged_terms = list(dict.fromkeys(keyword_terms + tag_tokens))

    fetch_effective = max(fetch_k, final_k * 2, 12)
    keyword_effective = max(keyword_k, 16)

    entries: dict[str, dict[str, Any]] = {}

    for q in rule_queries:
        sim_docs = store.similarity_search(q, k=max(fetch_effective, final_k))
        for rank, doc in enumerate(sim_docs):
            key = _doc_key(doc)
            e = entries.setdefault(key, {"doc": doc, "vector": 0.0, "mmr": 0.0, "keyword": 0.0})
            e["vector"] = max(float(e["vector"]), 1.0 / (1.0 + rank))

        mmr_docs = store.max_marginal_relevance_search(
            q,
            k=max(final_k + 3, min(final_k + 4, fetch_effective)),
            fetch_k=max(fetch_effective, final_k * 2),
        )
        for rank, doc in enumerate(mmr_docs):
            key = _doc_key(doc)
            e = entries.setdefault(key, {"doc": doc, "vector": 0.0, "mmr": 0.0, "keyword": 0.0})
            e["mmr"] = max(float(e["mmr"]), 1.0 / (1.0 + rank))

    if merged_terms and keyword_effective > 0:
        all_docs = _all_docs_from_store(store)
        scored: list[tuple[float, Document]] = []
        for doc in all_docs:
            s = _keyword_score(doc.page_content, merged_terms)
            if s > 0:
                scored.append((s, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        for s, doc in scored[:keyword_effective]:
            key = _doc_key(doc)
            e = entries.setdefault(key, {"doc": doc, "vector": 0.0, "mmr": 0.0, "keyword": 0.0})
            e["keyword"] = max(float(e["keyword"]), float(s))

    pool_size = max(final_k * 4, fetch_effective + 8)

    def pre_rank_score(e: dict[str, Any]) -> float:
        v, m, k = float(e["vector"]), float(e["mmr"]), float(e["keyword"])
        return 0.45 * v + 0.22 * m + 0.33 * k

    pool = sorted(entries.values(), key=pre_rank_score, reverse=True)[:pool_size]

    def final_score(e: dict[str, Any]) -> float:
        v, m, kw = float(e["vector"]), float(e["mmr"]), float(e["keyword"])
        base = 0.42 * v + 0.23 * m + 0.35 * kw
        doc = e["doc"]
        qmul = _rule_source_quality_multiplier(doc)
        tag_s = _tag_rule_match_score(doc, tags_list)
        forced = _tag_forced_path_boost(doc, tags_list)
        # Quality acts as a soft gate on shallow pages; tags and forced paths add targeted boost.
        return base * (0.38 + 0.62 * qmul) + 0.22 * tag_s + 0.30 * forced

    pool.sort(key=final_score, reverse=True)
    return [e["doc"] for e in pool[:final_k]]


def _doc_key(doc: Document) -> str:
    md = doc.metadata or {}
    source = str(md.get("source", ""))
    card_id = str(md.get("card_id", ""))
    rule_no = str(md.get("rule_number", ""))
    section_idx = str(md.get("section_index", ""))
    snippet = doc.page_content[:120].replace("\n", " ")
    return "|".join([source, card_id, rule_no, section_idx, snippet])


def _strip_json_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```\s*$", "", t)
    return t.strip()


def _parse_coverage_json(raw: str) -> dict[str, Any] | None:
    try:
        data = json.loads(_strip_json_fence(raw))
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def _merge_cap_docs(primary: list[Document], extra: list[Document], k: int) -> list[Document]:
    """Dedupe by identity; preserve order (primary first)."""
    out: list[Document] = []
    seen: set[str] = set()
    for doc in primary + extra:
        key = _doc_key(doc)
        if key in seen:
            continue
        seen.add(key)
        out.append(doc)
        if len(out) >= k:
            break
    return out


def _build_second_pass_queries(
    question: str,
    tags_list: list[str],
    base_queries: list[str],
    missing_evidence: str,
    reason: str,
) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for q in base_queries:
        q = q.strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    blob = f"{missing_evidence} {reason}".strip()
    for t in _tokenize(blob)[:28]:
        sq = f"{question} {t}".strip()
        if sq.lower() not in seen:
            seen.add(sq.lower())
            out.append(sq)
    if missing_evidence.strip():
        mq = f"{question} {missing_evidence.strip()[:400]}".strip()
        if mq.lower() not in seen:
            seen.add(mq.lower())
            out.append(mq)
    tag_join = " ".join(t.strip() for t in tags_list if t.strip())
    if tag_join:
        tq = f"{tag_join} {question}".strip()
        if tq.lower() not in seen:
            seen.add(tq.lower())
            out.append(tq)
    return out[:14]


def _evaluate_coverage(
    llm: ChatOpenAI,
    *,
    question: str,
    tags_context: str,
    card_context: str,
    rule_context: str,
) -> dict[str, Any]:
    user = (
        f"Question:\n{question}\n\n"
        f"Tags (metadata):\n{tags_context}\n\n"
        f"Retrieved cards:\n{card_context or '(none)'}\n\n"
        f"Retrieved rules:\n{rule_context or '(none)'}\n"
    )
    resp = llm.invoke(
        [
            SystemMessage(content=COVERAGE_SYSTEM_PROMPT),
            HumanMessage(content=user),
        ]
    )
    raw = str(resp.content).strip()
    parsed = _parse_coverage_json(raw)
    if parsed is None:
        return {
            "sufficient": True,
            "reason": "coverage parse failed; proceeding without second pass",
            "missing_evidence": "",
            "raw": raw,
        }
    sufficient = bool(parsed.get("sufficient", True))
    reason = str(parsed.get("reason", "") or "").strip()
    missing = str(parsed.get("missing_evidence", "") or "").strip()
    return {
        "sufficient": sufficient,
        "reason": reason,
        "missing_evidence": missing,
        "raw": raw,
    }


POLARITY_VERIFIER_SYSTEM_PROMPT = (
    "You verify whether a Yu-Gi-Oh! ruling sentence matches the decisive quoted rule/card text. "
    "Use ONLY the ruling line and decisive quote provided. Reply with JSON only:\n"
    '{"consistent": true or false, "reason": "one short sentence", '
    '"should_flip": true or false}. '
    "If the decisive quote clearly implies the opposite polarity of the ruling, set should_flip=true."
)


def _verify_polarity(llm: ChatOpenAI, ruling_line: str, decisive_quote: str) -> dict[str, Any]:
    user = (
        f"Ruling line:\n{ruling_line.strip()}\n\n"
        f"Decisive quote:\n{decisive_quote.strip()}\n"
    )
    resp = llm.invoke(
        [
            SystemMessage(content=POLARITY_VERIFIER_SYSTEM_PROMPT),
            HumanMessage(content=user),
        ]
    )
    raw = str(resp.content).strip()
    parsed = _parse_coverage_json(raw)
    if parsed is None:
        return {"consistent": True, "should_flip": False, "raw": raw}
    return {
        "consistent": bool(parsed.get("consistent", True)),
        "should_flip": bool(parsed.get("should_flip", False)),
        "raw": raw,
    }


_STORE_DOC_CACHE: dict[int, list[Document]] = {}
_CARD_NAME_INDEX_CACHE: dict[int, dict[str, Document]] = {}


def _all_docs_from_store(store: FAISS) -> list[Document]:
    key = id(store)
    cached = _STORE_DOC_CACHE.get(key)
    if cached is not None:
        return cached
    docs: list[Document] = []
    data = getattr(store.docstore, "_dict", {})
    for v in data.values():
        if isinstance(v, Document):
            docs.append(v)
    _STORE_DOC_CACHE[key] = docs
    return docs


def _normalize_card_name(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower())).strip()


def _build_card_name_index(store: FAISS) -> dict[str, Document]:
    key = id(store)
    cached = _CARD_NAME_INDEX_CACHE.get(key)
    if cached is not None:
        return cached

    index: dict[str, Document] = {}
    for doc in _all_docs_from_store(store):
        name = str(doc.metadata.get("card_name", "")).strip()
        if not name:
            name = _get_card_name(doc)
        norm = _normalize_card_name(name)
        if norm and norm not in index:
            index[norm] = doc

    _CARD_NAME_INDEX_CACHE[key] = index
    return index


def _extract_exact_card_docs(question: str, cards_store: FAISS, max_count: int) -> list[Document]:
    if max_count <= 0:
        return []

    index = _build_card_name_index(cards_store)
    found: list[Document] = []
    seen_doc_keys: set[str] = set()

    # First pass: quoted strings in question.
    for q in _extract_quoted_terms(question):
        norm_q = _normalize_card_name(q)
        doc = index.get(norm_q)
        if not doc:
            continue
        key = _doc_key(doc)
        if key not in seen_doc_keys:
            seen_doc_keys.add(key)
            found.append(doc)
        if len(found) >= max_count:
            return found

    # Second pass: n-gram exact matches against official card names.
    tokens = re.findall(r"[a-z0-9]+", question.lower())
    n = len(tokens)
    max_ngram = min(8, n)
    for width in range(max_ngram, 1, -1):
        for i in range(0, n - width + 1):
            phrase = " ".join(tokens[i : i + width]).strip()
            if not phrase:
                continue
            doc = index.get(phrase)
            if not doc:
                continue
            key = _doc_key(doc)
            if key in seen_doc_keys:
                continue
            seen_doc_keys.add(key)
            found.append(doc)
            if len(found) >= max_count:
                return found
    return found


def _keyword_score(text: str, terms: list[str]) -> float:
    if not terms:
        return 0.0
    low = text.lower()
    score = 0.0
    for term in terms:
        if not term:
            continue
        t = term.lower()
        if len(t) >= 3 and t in low:
            score += 1.0
        if " " in t and t in low:
            score += 1.0
    return score / max(len(terms), 1)


def _retrieve_candidates(
    *,
    store: FAISS,
    queries: list[str],
    final_k: int,
    fetch_k: int,
    keyword_k: int,
    keyword_terms: list[str],
) -> list[Document]:
    entries: dict[str, dict[str, Any]] = {}

    for q in queries:
        sim_docs = store.similarity_search(q, k=max(fetch_k, final_k))
        for rank, doc in enumerate(sim_docs):
            key = _doc_key(doc)
            e = entries.setdefault(key, {"doc": doc, "vector": 0.0, "mmr": 0.0, "keyword": 0.0})
            e["vector"] = max(float(e["vector"]), 1.0 / (1.0 + rank))

        mmr_docs = store.max_marginal_relevance_search(
            q,
            k=max(final_k, min(final_k + 2, fetch_k)),
            fetch_k=max(fetch_k, final_k * 2),
        )
        for rank, doc in enumerate(mmr_docs):
            key = _doc_key(doc)
            e = entries.setdefault(key, {"doc": doc, "vector": 0.0, "mmr": 0.0, "keyword": 0.0})
            e["mmr"] = max(float(e["mmr"]), 1.0 / (1.0 + rank))

    if keyword_terms and keyword_k > 0:
        all_docs = _all_docs_from_store(store)
        scored: list[tuple[float, Document]] = []
        for doc in all_docs:
            s = _keyword_score(doc.page_content, keyword_terms)
            if s > 0:
                scored.append((s, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        for s, doc in scored[:keyword_k]:
            key = _doc_key(doc)
            e = entries.setdefault(key, {"doc": doc, "vector": 0.0, "mmr": 0.0, "keyword": 0.0})
            e["keyword"] = max(float(e["keyword"]), float(s))

    ranked = sorted(
        entries.values(),
        key=lambda e: (0.5 * float(e["vector"])) + (0.25 * float(e["mmr"])) + (0.25 * float(e["keyword"])),
        reverse=True,
    )
    return [e["doc"] for e in ranked[:final_k]]


def ask_ygo_judge(
    question: str,
    tags: str | list[str] | None = None,
    rules_index_dir: str = "faiss_rules_index",
    cards_index_dir: str = "faiss_cards_index",
    qa_index_dir: str = "faiss_qa_index",
    embedding_model: str = "text-embedding-3-small",
    chat_model: str = "gpt-4o-mini",
    temperature: float = 0.1,
    use_rag: bool = True,
    rules_k: int = 5,
    cards_k: int = 5,
    qa_k: int = 2,
    fetch_k: int = 10,
    keyword_k: int = 12,
    coverage_check: bool = True,
    second_pass_on_gap: bool = True,
) -> dict[str, Any]:
    """
    Ask a YGO judge question. With use_rag=True, retrieval uses local FAISS indexes.

    Returns:
        {
            "answer": "...with references (RAG) or plain answer (no RAG)...",
            "references": [...],
            "cards": [Document, ...],
            "rules": [Document, ...],
            "qa": [Document, ...],
            "coverage": { ... }  (only when use_rag=True),
        }
    """
    if not question.strip():
        raise ValueError("question cannot be empty.")
    ensure_openai_api_key()

    tags_list = _normalize_tags(tags)
    tags_context = ", ".join(tags_list) if tags_list else "None provided."

    coverage_meta: dict[str, Any] = {}

    if use_rag:
        queries, topics, quoted_cards = _build_retrieval_queries(question, tags_list)
        rules_store = _load_faiss(index_dir=rules_index_dir, embedding_model=embedding_model)
        cards_store = _load_faiss(index_dir=cards_index_dir, embedding_model=embedding_model)
        try:
            qa_store = _load_faiss(index_dir=qa_index_dir, embedding_model=embedding_model)
        except FileNotFoundError:
            qa_store = None
        rule_terms = _tokenize(" ".join(queries + topics + tags_list + quoted_cards))
        cards_terms = _tokenize(" ".join(queries + quoted_cards + tags_list))
        qa_terms = _tokenize(" ".join(queries + tags_list + quoted_cards))

        cards = _retrieve_candidates(
            store=cards_store,
            queries=queries,
            final_k=cards_k,
            fetch_k=fetch_k,
            keyword_k=max(6, keyword_k // 2),
            keyword_terms=cards_terms + [x.lower() for x in quoted_cards],
        )
        exact_cards = _extract_exact_card_docs(question=question, cards_store=cards_store, max_count=cards_k)
        if exact_cards:
            merged_cards: list[Document] = []
            seen_keys: set[str] = set()
            for doc in exact_cards + cards:
                key = _doc_key(doc)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                merged_cards.append(doc)
                if len(merged_cards) >= cards_k:
                    break
            cards = merged_cards
        rules = _retrieve_rule_candidates(
            store=rules_store,
            question=question,
            base_queries=queries,
            tags_list=tags_list,
            final_k=rules_k,
            fetch_k=fetch_k,
            keyword_k=keyword_k,
            keyword_terms=rule_terms,
        )
        qa_docs: list[Document] = []
        if qa_store is not None and qa_k > 0:
            qa_docs = _retrieve_candidates(
                store=qa_store,
                queries=queries,
                final_k=qa_k,
                fetch_k=max(fetch_k, 10),
                keyword_k=max(6, keyword_k // 2),
                keyword_terms=qa_terms,
            )

        card_context = _format_context(cards, "CARD")
        rule_context = _format_context(rules, "RULE")
        qa_context = _format_context(qa_docs, "QA")
        coverage_rule_context = (
            f"{rule_context}\n\n{qa_context}" if qa_context else rule_context
        )

        coverage_meta = {
            "first_pass_sufficient": True,
            "first_pass_reason": "",
            "second_pass_used": False,
            "missing_evidence": "",
            "coverage_check_enabled": coverage_check,
        }

        if coverage_check:
            cov_llm = ChatOpenAI(model=chat_model, temperature=0.0)
            cov = _evaluate_coverage(
                cov_llm,
                question=question,
                tags_context=tags_context,
                card_context=card_context,
                rule_context=coverage_rule_context,
            )
            sufficient = bool(cov.get("sufficient", True))
            coverage_meta["first_pass_sufficient"] = sufficient
            coverage_meta["first_pass_reason"] = str(cov.get("reason", "") or "")
            coverage_meta["missing_evidence"] = str(cov.get("missing_evidence", "") or "")

            if (not sufficient) and second_pass_on_gap:
                miss = str(cov.get("missing_evidence", "") or "")
                reas = str(cov.get("reason", "") or "")
                sp_queries = _build_second_pass_queries(
                    question, tags_list, queries, miss, reas
                )
                extra_terms = _tokenize(f"{miss} {reas}")
                rule_terms_sp = list(dict.fromkeys(rule_terms + extra_terms))
                cards_terms_sp = list(dict.fromkeys(cards_terms + extra_terms))
                fetch_2 = min(fetch_k + 8, 28)
                kw_rules_2 = min(keyword_k + 10, 45)
                kw_cards_2 = max(8, (keyword_k // 2) + 6)

                rules_2 = _retrieve_rule_candidates(
                    store=rules_store,
                    question=question,
                    base_queries=sp_queries,
                    tags_list=tags_list,
                    final_k=rules_k,
                    fetch_k=fetch_2,
                    keyword_k=kw_rules_2,
                    keyword_terms=rule_terms_sp,
                )
                cards_2 = _retrieve_candidates(
                    store=cards_store,
                    queries=sp_queries,
                    final_k=cards_k,
                    fetch_k=fetch_2,
                    keyword_k=kw_cards_2,
                    keyword_terms=cards_terms_sp + [x.lower() for x in quoted_cards],
                )
                if exact_cards:
                    seed_c = _merge_cap_docs(exact_cards, cards, cards_k)
                    cards = _merge_cap_docs(seed_c, cards_2, cards_k)
                else:
                    cards = _merge_cap_docs(cards, cards_2, cards_k)
                rules = _merge_cap_docs(rules, rules_2, rules_k)
                if qa_store is not None and qa_k > 0:
                    qa_2 = _retrieve_candidates(
                        store=qa_store,
                        queries=sp_queries,
                        final_k=qa_k,
                        fetch_k=max(fetch_2, 10),
                        keyword_k=max(6, kw_cards_2),
                        keyword_terms=list(dict.fromkeys(qa_terms + extra_terms)),
                    )
                    qa_docs = _merge_cap_docs(qa_docs, qa_2, qa_k)
                coverage_meta["second_pass_used"] = True

                card_context = _format_context(cards, "CARD")
                rule_context = _format_context(rules, "RULE")
                qa_context = _format_context(qa_docs, "QA")
                coverage_rule_context = (
                    f"{rule_context}\n\n{qa_context}" if qa_context else rule_context
                )

        retrieval_note = ""
        if coverage_meta.get("second_pass_used"):
            retrieval_note = (
                "Retrieval note: A second retrieval pass ran after a pessimistic first-pass coverage check; "
                "snippets below are merged (deduplicated). If merged snippets clearly support a Yes/No ruling, "
                "you may answer decisively.\n\n"
            )
        elif coverage_check and (not coverage_meta.get("first_pass_sufficient", True)) and (
            not second_pass_on_gap
        ):
            retrieval_note = (
                "Coverage note: Initial retrieval looked incomplete and no second pass was allowed. "
                "Unless the snippets below clearly contain a decisive rule or card line, use exactly: "
                "\"Cannot determine from the provided snippets.\" for the ruling line.\n\n"
            )

        user_prompt = (
            retrieval_note
            + f"User question:\n{question}\n\n"
            f"Known question tags (metadata only):\n{tags_context}\n\n"
            f"Retrieved card data (up to {cards_k}):\n{card_context or 'No card data retrieved.'}\n\n"
            f"Retrieved rule passages (up to {rules_k}):\n{rule_context or 'No rule passages retrieved.'}\n\n"
            f"Retrieved prior official Q&A examples (up to {qa_k}):\n{qa_context or 'No Q&A examples retrieved.'}\n\n"
            "Output format (follow exactly):\n"
            "1) Ruling: one clear sentence starting with exactly one of: Yes / No / Can / Cannot. "
            "If the merged snippets still cannot support a decisive ruling, use exactly: "
            "\"Cannot determine from the provided snippets.\"\n"
            "2) Evidence synthesis: 2-5 bullets. You may synthesize multiple snippets to reach a conclusion, "
            "but every bullet must cite which snippets were combined (e.g. [RULE 2]+[CARD 1], or [QA 1]+[RULE 3]).\n"
            "3) Evidence plan: 2-5 bullets. Each bullet must cite one supporting snippet label such as "
            "[CARD 1] or [RULE 2], and explain what that snippet proves.\n"
            "4) PSCT analysis (only if relevant card text is present in retrieved card snippets):\n"
            "- For ':' and ';', FIRST check whether that symbol actually appears in any retrieved [CARD n] text.\n"
            "- If ':' appears, provide Condition (:) quote using the smallest exact substring around ':' from [CARD n].\n"
            "- If ';' appears, provide Cost/Procedure (;) quote using the smallest exact substring around ';' from [CARD n].\n"
            "- If ':' or ';' does not appear in retrieved [CARD n], write exactly: \"Absent in snippet\". "
            "Treat this as neutral, not an error.\n"
            "- Restriction quote (e.g. 'You can only...', 'once per turn'): smallest exact quote from [CARD n], "
            "or \"Absent in snippet\".\n"
            "- Mapping: one bullet for how the scenario satisfies the ':' condition (or \"N/A\" if absent), and one "
            "bullet for whether the ';' procedure/cost is possible/paid (or \"N/A\" if absent).\n"
            "Do not infer PSCT details that are not in retrieved card text; only quote from [CARD n].\n"
            "5) Missing evidence: if something is still missing after merged retrieval, say what is missing; "
            "otherwise write \"None\".\n\n"
            "Faithfulness rules:\n"
            "- Quote or closely paraphrase only from retrieved snippets.\n"
            "- Do not use outside knowledge.\n"
            "- If snippets conflict, say so and resolve using retrieved rule passages."
        )
        system_msg = SYSTEM_PROMPT
    else:
        cards: list[Document] = []
        rules: list[Document] = []
        user_prompt = (
            f"User question:\n{question}\n\n"
            f"Known question tags (metadata only):\n{tags_context}\n\n"
            "Provide a judge-style ruling and explain the reasoning clearly."
        )
        system_msg = SYSTEM_PROMPT_NO_RAG

    llm = ChatOpenAI(model=chat_model, temperature=temperature)
    response = llm.invoke(
        [
            SystemMessage(content=system_msg),
            HumanMessage(content=user_prompt),
        ]
    )

    answer_text = str(response.content).strip()

    if use_rag:
        references: list[str] = []
        for doc in cards:
            references.append(f"Card: {_get_card_name(doc)}")
        for doc in rules:
            references.append(f"Rule: {_get_rule_ref(doc)}")
        for doc in qa_docs:
            references.append(f"QA: {_get_qa_ref(doc)}")

        if not references:
            references.append("None")

        answer_with_refs = (
            f"{answer_text}\n\n"
            "References:\n"
            + "\n".join(f"- {item}" for item in references)
        )
    else:
        references = ["(no retrieval — model knowledge only)"]
        answer_with_refs = answer_text

    out: dict[str, Any] = {
        "answer": answer_with_refs,
        "references": references,
        "cards": cards,
        "rules": rules,
        "qa": qa_docs if use_rag else [],
    }
    if use_rag:
        out["coverage"] = coverage_meta
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask YGO judge questions from dual local FAISS indexes.")
    parser.add_argument("--question", required=True, help="User question for the YGO judge.")
    parser.add_argument(
        "--rules-index-dir",
        default="faiss_rules_index",
        help="Directory for the rules FAISS index.",
    )
    parser.add_argument(
        "--cards-index-dir",
        default="faiss_cards_index",
        help="Directory for the cards FAISS index.",
    )
    parser.add_argument(
        "--qa-index-dir",
        default="faiss_qa_index",
        help="Directory for the Q&A FAISS index.",
    )
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="Embedding model used for loading FAISS index.",
    )
    parser.add_argument("--chat-model", default="gpt-4o-mini", help="Chat model for final answer.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature.")
    parser.add_argument("--rules-k", type=int, default=5, help="Final number of rule chunks in context.")
    parser.add_argument("--cards-k", type=int, default=5, help="Final number of card chunks in context.")
    parser.add_argument("--qa-k", type=int, default=2, help="Final number of Q&A snippets in context.")
    parser.add_argument(
        "--fetch-k",
        type=int,
        default=10,
        help="Initial vector/MMR candidate count before dedupe and rerank.",
    )
    parser.add_argument(
        "--keyword-k",
        type=int,
        default=12,
        help="Keyword-based retrieval candidates (hybrid with vector/MMR).",
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Answer without FAISS retrieval (ablation / pure LLM).",
    )
    parser.add_argument(
        "--no-coverage-check",
        action="store_true",
        help="Skip LLM coverage audit and second retrieval pass.",
    )
    parser.add_argument(
        "--no-second-pass",
        action="store_true",
        help="Run coverage check but do not retrieve again if insufficient.",
    )
    args = parser.parse_args()

    result = ask_ygo_judge(
        question=args.question,
        rules_index_dir=args.rules_index_dir,
        cards_index_dir=args.cards_index_dir,
        qa_index_dir=args.qa_index_dir,
        embedding_model=args.embedding_model,
        chat_model=args.chat_model,
        temperature=args.temperature,
        use_rag=not args.no_rag,
        rules_k=args.rules_k,
        cards_k=args.cards_k,
        qa_k=args.qa_k,
        fetch_k=args.fetch_k,
        keyword_k=args.keyword_k,
        coverage_check=not args.no_coverage_check,
        second_pass_on_gap=not args.no_second_pass,
    )
    print(result["answer"])


if __name__ == "__main__":
    main()
