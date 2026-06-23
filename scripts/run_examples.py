from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fashion_assistant import FashionRecommender, load_fashion_data  # noqa: E402


EXAMPLES = [
    "I need an outfit for a business meeting.",
    "Suggest a smart casual outfit for a dinner date.",
    "I am attending a wedding next weekend.",
    "I am a 22-year-old male looking for a casual summer outfit.",
]


def main() -> None:
    recommender = FashionRecommender(load_fashion_data(ROOT))

    for query in EXAMPLES:
        print("\n" + "=" * 80)
        print("User:", query)
        recommendations = recommender.recommend_from_chat(query, top_k=1)
        if not recommendations:
            print("No recommendation found.")
            continue

        recommendation = recommendations[0]
        print("Recommendation:", recommendation.title)
        print("Score:", round(recommendation.score, 3))
        for item in recommendation.items:
            print(f"- {item.role_label}: {item.name} ({item.color}, {item.occasion})")
        print("Reason:", recommendation.reason)


if __name__ == "__main__":
    main()
