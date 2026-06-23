from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fashion_assistant import FashionRecommender, load_fashion_data  # noqa: E402


def main() -> None:
    data = load_fashion_data(ROOT)
    recommender = FashionRecommender(data)
    summary = recommender.dataset_summary()

    print("Dare XAI Fashion Dataset Summary")
    print("=" * 34)
    print(f"Products: {summary['product_count']}")
    print(f"Curated outfits: {summary['outfit_count']}")
    print("\nGender counts")
    for key, value in summary["gender_counts"].items():
        print(f"- {key}: {value}")
    print("\nOccasion counts")
    for key, value in summary["occasion_counts"].items():
        print(f"- {key}: {value}")
    print("\nRole counts")
    for key, value in summary["role_counts"].items():
        print(f"- {key}: {value}")
    print("\nTop categories")
    for key, value in summary["category_counts"].items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
