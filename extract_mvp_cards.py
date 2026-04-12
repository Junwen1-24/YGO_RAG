import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent
CARD_JSON_PATH = PROJECT_ROOT / "data" / "card.json"
OUTPUT_PATH = PROJECT_ROOT / "mvp_cards.json"


MVP_CARD_NAMES = [
    # Hand traps / monsters / spells / traps, using official English names
    "Ash Blossom & Joyous Spring",
    "Maxx \"C\"",
    "Infinite Impermanence",
    "Effect Veiler",
    "Nibiru, the Primal Being",
    "Called by the Grave",
    "Ghost Belle & Haunted Mansion",
    "Droll & Lock Bird",
    "Ghost Mourner & Moonlit Chill",
    "Skull Meister",
    "Dimension Shifter",
    "Artifact Lancea",
    "PSY-Framegear Gamma",
    "Ghost Sister & Spooky Dogwood",
    "D.D. Crow",
    "Kuriboh",
    "Honest",
    "Fantastical Dragon Phantazmay",
    "Token Collector",
    "Crossout Designator",
    "Sauravis, the Ancient and Ascended",
    "Herald of Orange Light",
    "Retaliating \"C\"",
    "Contact \"C\"",
    "Gnomaterial",
    "Monster Reborn",
    "Pot of Prosperity",
    "Forbidden Droplet",
    "Super Polymerization",
    "Raigeki",
    "Dark Ruler No More",
    "Triple Tactics Talent",
    "Solemn Judgment",
    "Skill Drain",
    "Evenly Matched",
    "Mystical Space Typhoon",
    "Cosmic Cyclone",
    "Harpie's Feather Duster",
    "Rivalry of Warlords",
    "Gozen Match",
    "There Can Be Only One",
    "Imperial Order",
    "Anti-Spell Fragrance",
    "Foolish Burial",
    "Terraforming",
    "Dupe Frog",
    "Peten the Dark Clown",
    "Elemental HERO Stratos",
    "Jiaotu, Darkness of the Yang Zing",
    "Geartown",
    "Lightpulsar Dragon",
    "Dark Magician",
    "Blue-Eyes White Dragon",
    "Red-Eyes B. Dragon",
    "Jinzo",
    "Royal Decree",
    "Vanity's Emptiness",
    "Macro Cosmos",
    "Dimensional Fissure",
    "Torrential Tribute",
    "Accesscode Talker",
    "Baronne de Fleur",
    "Divine Arsenal AA-ZEUS - Sky Thunder",
    "Crystron Halqifibrax",
    "Predaplant Verte Anaconda",
    "Apollousa, Bow of the Goddess",
    "I:P Masquerena",
    "Knightmare Phoenix",
    "Knightmare Unicorn",
    "Borreload Savage Dragon",
    "Elder Entity N'tss",
    "Invoked Mechaba",
    "Mirrorjade the Iceblade Dragon",
    "Destiny HERO - Destroyer Phoenix Enforcer",
    "Thunder Dragon Colossus",
    "Salamangreat Almiraj",
    "Linkuriboh",
    "Heavymetalfoes Electrumite",
    "Saryuja Skull Dread",
    "Striker Dragon",
    "Exodia the Forbidden One",
    "Last Turn",
    "Pole Position",
    "Convulsion of Nature",
    "Interrupted Kaiju Slumber",
    "Gameciel, the Sea Turtle Kaiju",
    "Lava Golem",
    "The Winged Dragon of Ra - Sphere Mode",
    "Inspector Boarder",
    "Mystic Mine",
    "Eater of Millions",
    "Gren Maju Da Eiza",
    "Sky Striker Ace - Shizuku",
    "Sky Striker Ace - Raye",
    "Sky Striker Mecha - Multirole",
    "Eldlich the Golden Lord",
    "Floowandereeze & Empen",
    "Tearlaments Kitkallos",
    "Spright Elf",
    "Kashtira Fenrir",
]


def main() -> None:
    if not CARD_JSON_PATH.exists():
        raise FileNotFoundError(f"Card database not found: {CARD_JSON_PATH}")

    with CARD_JSON_PATH.open("r", encoding="utf-8") as f:
        db = json.load(f)

    cards = db.get("data", [])

    # Build index by exact English name (case-sensitive to match API)
    index = {}
    for card in cards:
        name = card.get("name")
        if not name:
            continue
        index.setdefault(name, []).append(card)

    selected = []
    missing = []

    for name in MVP_CARD_NAMES:
        matches = index.get(name)
        if not matches:
            missing.append(name)
            continue
        # If multiple printings exist, keep them all
        selected.extend(matches)

    OUTPUT_PATH.write_text(
        json.dumps({"data": selected, "missing_names": missing}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved {len(selected)} cards to {OUTPUT_PATH}")
    if missing:
        print("Missing names (not found in card.json):")
        for n in missing:
            print(f"- {n}")


if __name__ == "__main__":
    main()

