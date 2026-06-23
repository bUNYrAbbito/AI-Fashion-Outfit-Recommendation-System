from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fashion_assistant import FashionRecommender, load_fashion_data, parse_user_message  # noqa: E402


def build_recommender() -> FashionRecommender:
    return FashionRecommender(load_fashion_data(ROOT))


def test_intent_parser_detects_profile_details() -> None:
    intent = parse_user_message("I am a 22-year-old male looking for a casual summer outfit.")
    assert intent.gender == "men"
    assert intent.age == 22
    assert intent.occasion in {"casual", "vacation"}


def test_business_meeting_recommendation_has_complete_roles() -> None:
    recommender = build_recommender()
    recommendations = recommender.recommend_from_chat(
        "I need an outfit for a business meeting.",
        profile={"gender": "men", "occasion": "office", "style": "formal"},
        top_k=1,
    )
    assert recommendations
    roles = {item.role for item in recommendations[0].items}
    assert "footwear" in roles
    assert "topwear" in roles or "main" in roles
    assert "bottomwear" in roles or "main" in roles


def test_product_search_can_build_around_white_shirt() -> None:
    recommender = build_recommender()
    results = recommender.search_products("white formal shirt", gender="men", top_k=3)
    assert not results.empty

    product_id = str(results.iloc[0]["id"])
    recommendation = recommender.recommend_around_product(product_id)
    assert recommendation is not None
    assert len(recommendation.items) >= 3
