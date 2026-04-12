import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
HTML_TAG_RE = re.compile(r"<[^>]+>")
SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
RULE_NUMBER_RE = re.compile(r"^\s*((?:\d+\.)+\d*|\d+)")


def ensure_openai_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if api_key:
        return api_key
    raise EnvironmentError(
        "OPENAI_API_KEY is missing. Set it in the process environment, for example: "
        "export OPENAI_API_KEY='sk-...' (do not commit keys to the repository)."
    )


def iter_rule_files(rules_dir: Path) -> Iterable[Path]:
    for path in rules_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".md", ".html", ".htm", ".txt", ".json"}:
            continue
        yield path


def clean_html_text(raw: str) -> str:
    text = SCRIPT_STYLE_RE.sub(" ", raw)
    text = HTML_COMMENT_RE.sub(" ", text)
    text = re.sub(r"</(p|div|section|article|li|tr|h[1-6])\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = HTML_TAG_RE.sub(" ", text)
    return normalize_whitespace(text)


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_markdown_sections(text: str) -> List[tuple[str, str]]:
    matches = list(HEADING_RE.finditer(text))
    if not matches:
        cleaned = normalize_whitespace(text)
        return [("document", cleaned)] if cleaned else []

    sections: List[tuple[str, str]] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        heading = match.group(2).strip()
        body = normalize_whitespace(text[start:end])
        if body:
            sections.append((heading, body))
    return sections


def load_action_treated_glossary_json(data: dict[str, Any], rel_source: str) -> List[Document]:
    """Glossary matrices where each cell is { interaction: bool, references: [...] }."""
    docs: List[Document] = []
    for outer_key, inner_obj in data.items():
        if not isinstance(inner_obj, dict):
            continue
        for inner_key, cell in inner_obj.items():
            if not isinstance(cell, dict):
                continue
            if not isinstance(cell.get("interaction"), bool):
                continue
            refs = cell.get("references") or []
            refs_text = clean_html_text(" ".join(str(r) for r in refs)) if refs else ""
            flag = "yes" if cell["interaction"] else "no"
            body = normalize_whitespace(
                f'Glossary (action treated as): "{outer_key}" vs "{inner_key}" — '
                f'counts as the same kind of action for rule purposes: {flag}.\n'
                f"References:\n{refs_text}"
            )
            if not body:
                continue
            docs.append(
                Document(
                    page_content=body,
                    metadata={
                        "source": rel_source,
                        "doc_type": "rule",
                        "section_title": f"{outer_key} / {inner_key}",
                        "section_index": len(docs),
                        "glossary_kind": "action_treated_as",
                    },
                )
            )
    return docs


def load_effect_interaction_glossary_json(data: dict[str, Any], rel_source: str) -> List[Document]:
    """Glossary where each topic maps to a list of { interaction, example, reference }."""
    docs: List[Document] = []
    for category, topics in data.items():
        if not isinstance(topics, dict):
            continue
        for topic, entries in topics.items():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                inter = clean_html_text(str(entry.get("interaction", "")))
                ex = clean_html_text(str(entry.get("example", "")))
                ref = clean_html_text(str(entry.get("reference", "")))
                body = normalize_whitespace(
                    f"Glossary (effect interaction) — {category} / {topic}\n"
                    f"Ruling: {inter}\n"
                    f"Example: {ex}\n"
                    f"Reference: {ref}"
                )
                if not body.strip():
                    continue
                docs.append(
                    Document(
                        page_content=body,
                        metadata={
                            "source": rel_source,
                            "doc_type": "rule",
                            "section_title": f"{category} / {topic}",
                            "section_index": len(docs),
                            "glossary_kind": "effect_interaction",
                        },
                    )
                )
    return docs


def load_json_rule_docs(file_path: Path, rules_dir: Path) -> List[Document]:
    rel_source = str(file_path.relative_to(rules_dir.parent))
    raw = file_path.read_text(encoding="utf-8", errors="ignore")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []

    if not isinstance(data, dict):
        return []

    name = file_path.name.lower()
    if "action-treated-as-or-not" in name:
        return load_action_treated_glossary_json(data, rel_source)
    if "effect-interaction" in name:
        return load_effect_interaction_glossary_json(data, rel_source)

    # Unknown JSON under rules: single searchable blob
    compact = normalize_whitespace(json.dumps(data, ensure_ascii=False)[:200_000])
    if not compact:
        return []
    return [
        Document(
            page_content=compact,
            metadata={
                "source": rel_source,
                "doc_type": "rule",
                "section_title": file_path.stem,
                "section_index": 0,
                "glossary_kind": "json_blob",
            },
        )
    ]


def load_rule_docs(rules_dir: Path) -> List[Document]:
    docs: List[Document] = []
    for file_path in iter_rule_files(rules_dir):
        rel_source = str(file_path.relative_to(rules_dir.parent))

        if file_path.suffix.lower() == ".json":
            docs.extend(load_json_rule_docs(file_path, rules_dir))
            continue

        raw = file_path.read_text(encoding="utf-8", errors="ignore")

        if file_path.suffix.lower() in {".html", ".htm"}:
            raw = clean_html_text(raw)
            sections = [("document", raw)] if raw else []
        else:
            sections = split_markdown_sections(raw)

        for section_index, (section_title, section_text) in enumerate(sections):
            docs.append(
                Document(
                    page_content=section_text,
                    metadata={
                        "source": rel_source,
                        "doc_type": "rule",
                        "section_title": section_title,
                        "section_index": section_index,
                    },
                )
            )
    return docs


def load_card_docs(cards_json_path: Path) -> List[Document]:
    with cards_json_path.open(encoding="utf-8") as f:
        payload = json.load(f)

    card_rows = payload.get("data", payload if isinstance(payload, list) else [])
    if not isinstance(card_rows, list):
        raise ValueError("cards json must be a list or an object with a 'data' list.")

    docs: List[Document] = []
    for idx, card in enumerate(card_rows):
        if not isinstance(card, dict):
            continue

        name = str(card.get("name", "")).strip() or "Unknown Card"
        card_id = card.get("id")
        card_type = card.get("type", "")
        desc = card.get("desc", "")
        race = card.get("race", "")
        attribute = card.get("attribute", "")
        level = card.get("level", "")
        atk = card.get("atk", "")
        defense = card.get("def", "")
        archetype = card.get("archetype", "")

        lines = [
            f"name: {name}",
            f"id: {card_id}",
            f"type: {card_type}",
            f"race: {race}",
            f"attribute: {attribute}",
            f"level: {level}",
            f"atk: {atk}",
            f"def: {defense}",
            f"archetype: {archetype}",
            f"desc: {desc}",
        ]
        text = normalize_whitespace("\n".join(lines))

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(cards_json_path),
                    "doc_type": "card",
                    "card_index": idx,
                    "card_id": card_id,
                    "card_name": name,
                    "card_type": card_type,
                },
            )
        )

    return docs


def load_qa_docs(qa_csv_path: Path) -> List[Document]:
    docs: List[Document] = []
    with qa_csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    for idx, row in enumerate(rows):
        question = (row.get("Question") or "").strip()
        official = (row.get("Official_Answer") or "").strip()
        if not question or not official:
            continue
        tags = (row.get("Tags") or "").strip()
        row_id = (row.get("ID") or "").strip()
        text = normalize_whitespace(
            f"question: {question}\n"
            f"official_answer: {official}\n"
            f"tags: {tags or 'none'}"
        )
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(qa_csv_path),
                    "doc_type": "qa",
                    "qa_index": idx,
                    "qa_id": row_id,
                    "tags": tags,
                    "question": question,
                },
            )
        )
    return docs


def split_rule_docs(rule_docs: Iterable[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(list(rule_docs))


def annotate_rule_references(rule_chunks: Iterable[Document]) -> List[Document]:
    annotated: List[Document] = []
    for doc in rule_chunks:
        section_title = str(doc.metadata.get("section_title", ""))
        content_first_line = doc.page_content.splitlines()[0] if doc.page_content.strip() else ""
        probe = section_title if section_title else content_first_line
        match = RULE_NUMBER_RE.match(probe)
        doc.metadata["rule_number"] = match.group(1) if match else ""
        annotated.append(doc)
    return annotated


def build_faiss_index(
    docs: Iterable[Document],
    index_output_dir: Path,
    embedding_model: str = "text-embedding-3-small",
) -> None:
    all_docs = list(docs)
    if not all_docs:
        raise ValueError(f"No documents found for index output: {index_output_dir}")

    embeddings = OpenAIEmbeddings(model=embedding_model)
    vector_store = FAISS.from_documents(all_docs, embeddings)
    index_output_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(index_output_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dual FAISS indexes for YGO rules and cards.")
    parser.add_argument(
        "--rules-dir",
        default="ygo_rules",
        help="Directory containing rules files (.md/.html/.htm/.txt/.json)",
    )
    parser.add_argument(
        "--cards-json",
        default="data/card.json",
        help="Full card DB JSON (list or {'data': [...]}). Default builds from data/card.json, not mvp_cards.json.",
    )
    parser.add_argument(
        "--cards-only",
        action="store_true",
        help="Embed only the card index (skip rules; faster when rebuilding after card DB updates).",
    )
    parser.add_argument(
        "--rules-only",
        action="store_true",
        help="Embed only the rules index (skip cards).",
    )
    parser.add_argument(
        "--rules-index-dir",
        default="faiss_rules_index",
        help="Output directory for rules FAISS index",
    )
    parser.add_argument(
        "--cards-index-dir",
        default="faiss_cards_index",
        help="Output directory for cards FAISS index",
    )
    parser.add_argument(
        "--qa-csv",
        default="data/ygo_eval_dataset.csv",
        help="CSV with Question/Official_Answer used to build Q&A retrieval index.",
    )
    parser.add_argument(
        "--qa-index-dir",
        default="faiss_qa_index",
        help="Output directory for Q&A FAISS index.",
    )
    parser.add_argument(
        "--no-qa",
        action="store_true",
        help="Skip building the Q&A index.",
    )
    parser.add_argument("--chunk-size", type=int, default=800, help="Chunk size for text splitting")
    parser.add_argument("--chunk-overlap", type=int, default=120, help="Chunk overlap for text splitting")
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="OpenAI embedding model name",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_openai_api_key()

    if args.cards_only and args.rules_only:
        raise ValueError("Choose at most one of --cards-only and --rules-only.")

    build_rules = not args.cards_only
    build_cards = not args.rules_only
    build_qa = (not args.cards_only) and (not args.rules_only) and (not args.no_qa)

    rules_dir = Path(args.rules_dir)
    cards_json = Path(args.cards_json)
    rules_index_dir = Path(args.rules_index_dir)
    cards_index_dir = Path(args.cards_index_dir)
    qa_csv = Path(args.qa_csv)
    qa_index_dir = Path(args.qa_index_dir)

    if build_rules:
        if not rules_dir.exists() or not rules_dir.is_dir():
            raise FileNotFoundError(f"Rules directory not found: {rules_dir}")
    if build_cards:
        if not cards_json.exists() or not cards_json.is_file():
            raise FileNotFoundError(f"Cards JSON file not found: {cards_json}")
    if build_qa:
        if not qa_csv.exists() or not qa_csv.is_file():
            raise FileNotFoundError(f"Q&A CSV file not found: {qa_csv}")

    if build_rules:
        rule_docs = load_rule_docs(rules_dir)
        rule_chunks = split_rule_docs(rule_docs, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
        rule_chunks = annotate_rule_references(rule_chunks)
        build_faiss_index(rule_chunks, rules_index_dir, embedding_model=args.embedding_model)
        print(f"Loaded {len(rule_docs)} rule sections.")
        print(f"Created {len(rule_chunks)} rule chunks.")
        print(f"Rules index saved to: {rules_index_dir.resolve()}")

    if build_cards:
        card_docs = load_card_docs(cards_json)
        build_faiss_index(card_docs, cards_index_dir, embedding_model=args.embedding_model)
        print(f"Loaded {len(card_docs)} card docs from {cards_json}.")
        print(f"Cards index saved to: {cards_index_dir.resolve()}")

    if build_qa:
        qa_docs = load_qa_docs(qa_csv)
        build_faiss_index(qa_docs, qa_index_dir, embedding_model=args.embedding_model)
        print(f"Loaded {len(qa_docs)} QA docs from {qa_csv}.")
        print(f"QA index saved to: {qa_index_dir.resolve()}")


if __name__ == "__main__":
    main()
