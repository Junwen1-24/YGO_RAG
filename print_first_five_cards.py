#!/usr/bin/env python3
"""Load YGOPRODeck cardinfo JSON and print the first five cards."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    json_path = root / "ygoprodeck_cardinfo_full.json"

    with json_path.open(encoding="utf-8") as f:
        payload = json.load(f)

    cards = payload["data"]
    print(f"Total cards: {len(cards)}\n")

    for i, card in enumerate(cards[:5], start=1):
        print(f"--- Card {i} ---")
        print(f"id:   {card.get('id')}")
        print(f"name: {card.get('name')}")
        print(f"type: {card.get('type')} ({card.get('humanReadableCardType')})")
        print(f"keys: {sorted(card.keys())}")
        print(f"desc:\n{card.get('desc', '')}\n")


if __name__ == "__main__":
    main()
