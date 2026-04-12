#!/usr/bin/env python3
"""Open the Speed Duel rulebook PDF with PyMuPDF."""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF


def main() -> None:
    root = Path(__file__).resolve().parent
    pdf_path = root / "data" / "SD_RuleBook_EN_10.pdf"

    doc = fitz.open(pdf_path)
    try:
        print(f"Opened: {pdf_path}")
        print(f"Pages: {doc.page_count}")
        print(f"Metadata: {doc.metadata}")
    finally:
        doc.close()


if __name__ == "__main__":
    main()
