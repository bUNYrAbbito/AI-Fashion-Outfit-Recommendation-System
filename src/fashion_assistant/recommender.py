"""Hybrid outfit recommendation engine.

The engine combines three beginner-friendly ideas:

1. TF-IDF retrieval over product metadata to understand the user's text.
2. A compatibility graph learned from the 25 curated outfits.
3. Simple metadata and color rules to fill missing outfit roles.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .data_loader import FashionData
from .features import average_harmony
from .intent import UserIntent, merge_profile, parse_user_message
from .llm import improve_explanation_with_gemini


OUTFIT_ID_COLUMNS = [
    "hero_id",
    "second_id",
    "layer_id",
    "footwear_id",
    "accessory_1_id",
    "accessory_2_id",
]

ROLE_LABELS = {
    "main": "Main item",
    "topwear": "Topwear",
    "bottomwear": "Bottomwear",
    "layer": "Layer",
    "footwear": "Footwear",
    "accessory": "Accessory",
}


@dataclass
class RecommendedItem:
    id: str
    name: str
    brand: str
    role: str
    category: str
    occasion: str
    color: str
    price_inr: float
    rating: float | None
    image_path: str
    product_url: str

    @property
    def role_label(self) -> str:
        return ROLE_LABELS.get(self.role, self.role.title())


@dataclass
class OutfitRecommendation:
    title: str
    score: float
    items: list[RecommendedItem]
    reason: str
    source: str
    palette: str = ""

    @property
    def total_price_inr(self) -> float:
        return sum(item.price_inr for item in self.items if item.price_inr)


class FashionRecommender:
    def __init__(self, data: FashionData):
        self.data = data
        self.products = data.products.copy()
        self.outfits = data.outfits.copy()
        self.product_by_id = self.products.set_index("id", drop=False)

        self.product_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.product_matrix = self.product_vectorizer.fit_transform(self.products["search_text"])

        self.outfit_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.outfit_matrix = self.outfit_vectorizer.fit_transform(self.outfits["outfit_text"])

        self.compatibility_graph = self._build_compatibility_graph()

    def recommend_from_chat(
        self,
        message: str,
        profile: dict | None = None,
        top_k: int = 3,
        use_llm: bool = False,
    ) -> list[OutfitRecommendation]:
        intent = merge_profile(parse_user_message(message), profile)
        return self.recommend_for_intent(intent, top_k=top_k, use_llm=use_llm)

    def recommend_for_intent(
        self,
        intent: UserIntent,
        top_k: int = 3,
        use_llm: bool = False,
    ) -> list[OutfitRecommendation]:
        hero_candidates = self._hero_candidates(intent, limit=max(top_k * 4, 8))
        recommendations: list[OutfitRecommendation] = []
        seen_signatures: set[tuple[str, ...]] = set()

        for hero_id, hero_score in hero_candidates:
            recommendation = self.recommend_around_product(
                hero_id,
                intent=intent,
                base_score=hero_score,
                use_llm=use_llm,
            )
            if not recommendation:
                continue
            signature = tuple(sorted(item.id for item in recommendation.items))
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            recommendations.append(recommendation)

        if len(recommendations) < top_k:
            recommendations.extend(
                self._curated_outfit_recommendations(
                    intent,
                    limit=top_k - len(recommendations),
                    seen_signatures=seen_signatures,
                    use_llm=use_llm,
                )
            )

        recommendations.sort(key=lambda rec: rec.score, reverse=True)
        return recommendations[:top_k]

    def recommend_around_product(
        self,
        product_id: str,
        intent: UserIntent | None = None,
        base_score: float = 1.0,
        use_llm: bool = False,
    ) -> OutfitRecommendation | None:
        if product_id not in self.product_by_id.index:
            return None

        if intent is None:
            product = self.product_by_id.loc[product_id]
            intent = UserIntent(
                message=str(product["name"]),
                gender=str(product["gender"] or "") or None,
                occasion=str(product["occasion"] or "") or None,
                wear_type=str(product["wear_type"] or "") or None,
            )

        selected_ids = [product_id]
        hero = self.product_by_id.loc[product_id]
        needed_roles = self._needed_roles_for_anchor(hero, intent)

        for role in needed_roles:
            candidate_id = self._best_candidate_for_role(role, selected_ids, intent)
            if candidate_id:
                selected_ids.append(candidate_id)

        items = [self._row_to_item(self.product_by_id.loc[item_id]) for item_id in selected_ids]
        if not self._has_minimum_complete_outfit(items):
            return None

        graph_bonus = self._average_graph_strength(selected_ids)
        role_bonus = len({item.role for item in items}) * 0.35
        score = float(base_score + graph_bonus + role_bonus)
        title = self._make_title(items, intent)
        reason = self._build_reason(items, intent, graph_bonus)

        if use_llm:
            reason = improve_explanation_with_gemini(
                intent.message,
                self._items_summary(items),
                reason,
            )

        return OutfitRecommendation(
            title=title,
            score=score,
            items=items,
            reason=reason,
            source="Generated by TF-IDF + compatibility graph",
            palette=", ".join(dict.fromkeys(item.color for item in items if item.color)),
        )

    def search_products(
        self,
        query: str,
        gender: str | None = None,
        occasion: str | None = None,
        top_k: int = 8,
    ) -> pd.DataFrame:
        query_vector = self.product_vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.product_matrix).ravel()
        scored = self.products.copy()
        scored["score"] = similarities

        if gender and gender != "any":
            scored = scored[scored["gender"].eq(gender)]
            scored["score"] = scored["score"] + 0.2
        if occasion and occasion != "any":
            scored["score"] = scored["score"] + scored["occasion"].eq(occasion).astype(float) * 0.25

        return scored.sort_values("score", ascending=False).head(top_k)

    def dataset_summary(self) -> dict:
        return {
            "product_count": len(self.products),
            "outfit_count": len(self.outfits),
            "gender_counts": self.products["gender"].value_counts().to_dict(),
            "occasion_counts": self.products["occasion"].value_counts().to_dict(),
            "role_counts": self.products["role"].value_counts().to_dict(),
            "category_counts": self.products["category_label"].value_counts().head(12).to_dict(),
        }

    def _build_compatibility_graph(self) -> dict[str, dict[str, float]]:
        graph: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for _, outfit in self.outfits.iterrows():
            item_ids = [str(outfit[col]) for col in OUTFIT_ID_COLUMNS if str(outfit[col]).strip()]
            for left in item_ids:
                for right in item_ids:
                    if left != right:
                        graph[left][right] += 1.0
        return graph

    def _hero_candidates(self, intent: UserIntent, limit: int) -> list[tuple[str, float]]:
        query = intent.search_query()
        query_vector = self.product_vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.product_matrix).ravel()

        candidates: list[tuple[str, float]] = []
        for index, row in self.products.iterrows():
            role = row["role"]
            if role not in {"main", "topwear", "bottomwear"}:
                continue

            score = float(similarities[index])
            score += self._intent_match_score(row, intent)
            if role in {"main", "topwear"}:
                score += 0.3
            candidates.append((row["id"], score))

        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates[:limit]

    def _needed_roles_for_anchor(self, anchor: pd.Series, intent: UserIntent) -> list[str]:
        role = anchor["role"]
        category = anchor["category"]

        if role == "topwear":
            roles = ["bottomwear", "footwear", "accessory"]
        elif role == "bottomwear":
            roles = ["topwear", "footwear", "accessory"]
        elif role == "footwear":
            roles = ["topwear", "bottomwear", "accessory"]
        elif role == "layer":
            roles = ["topwear", "bottomwear", "footwear"]
        else:
            roles = ["footwear", "accessory"]
            if category == "suits":
                roles.insert(0, "topwear")

        if intent.occasion in {"office", "winter"} and "layer" not in roles and role != "main":
            roles.append("layer")
        return roles

    def _best_candidate_for_role(
        self,
        role: str,
        selected_ids: list[str],
        intent: UserIntent,
    ) -> str | None:
        candidates = self.products[self.products["role"].eq(role)]
        if candidates.empty:
            return None

        selected_rows = [self.product_by_id.loc[item_id] for item_id in selected_ids]
        selected_colors = [str(row["color"]) for row in selected_rows]
        query_vector = self.product_vectorizer.transform([intent.search_query()])
        candidate_vectors = self.product_matrix[candidates.index]
        similarities = cosine_similarity(query_vector, candidate_vectors).ravel()

        best_id = None
        best_score = -999.0
        for local_index, (_, row) in enumerate(candidates.iterrows()):
            product_id = row["id"]
            if product_id in selected_ids:
                continue

            score = float(similarities[local_index])
            score += self._intent_match_score(row, intent)
            score += self._graph_score(product_id, selected_ids) * 2.5
            score += average_harmony(str(row["color"]), selected_colors) * 0.7
            score += self._quality_score(row)

            if intent.budget_inr:
                current_total = sum(float(self.product_by_id.loc[item_id]["price_inr"] or 0) for item_id in selected_ids)
                if current_total + float(row["price_inr"] or 0) > intent.budget_inr:
                    score -= 1.0

            if score > best_score:
                best_id = str(product_id)
                best_score = score

        return best_id

    def _curated_outfit_recommendations(
        self,
        intent: UserIntent,
        limit: int,
        seen_signatures: set[tuple[str, ...]],
        use_llm: bool,
    ) -> list[OutfitRecommendation]:
        if limit <= 0:
            return []

        query_vector = self.outfit_vectorizer.transform([intent.search_query()])
        similarities = cosine_similarity(query_vector, self.outfit_matrix).ravel()
        scored_outfits = []
        for index, outfit in self.outfits.iterrows():
            score = float(similarities[index])
            if intent.gender and outfit["gender"] == intent.gender:
                score += 0.8
            if intent.occasion and outfit["occasion"] == intent.occasion:
                score += 1.0
            if intent.wear_type and outfit["wear_type"] == intent.wear_type:
                score += 0.5
            scored_outfits.append((index, score))

        scored_outfits.sort(key=lambda item: item[1], reverse=True)
        recommendations: list[OutfitRecommendation] = []
        for index, score in scored_outfits:
            outfit = self.outfits.loc[index]
            item_ids = [str(outfit[col]) for col in OUTFIT_ID_COLUMNS if str(outfit[col]).strip()]
            signature = tuple(sorted(item_ids))
            if signature in seen_signatures:
                continue
            items = [self._row_to_item(self.product_by_id.loc[item_id]) for item_id in item_ids if item_id in self.product_by_id.index]
            reason = str(outfit["stylist_rationale"])
            if use_llm:
                reason = improve_explanation_with_gemini(intent.message, self._items_summary(items), reason)
            recommendations.append(
                OutfitRecommendation(
                    title=str(outfit["theme"]),
                    score=float(score),
                    items=items,
                    reason=reason,
                    source=f"Curated outfit reference: {outfit['outfit_id']}",
                    palette=str(outfit["palette"]),
                )
            )
            seen_signatures.add(signature)
            if len(recommendations) >= limit:
                break
        return recommendations

    def _intent_match_score(self, row: pd.Series, intent: UserIntent) -> float:
        score = 0.0
        if intent.gender and row["gender"] == intent.gender:
            score += 1.0
        elif intent.gender and row["gender"] != intent.gender:
            score -= 2.0

        if intent.occasion and row["occasion"] == intent.occasion:
            score += 0.9
        elif intent.occasion and row["occasion"] in self._nearby_occasions(intent.occasion):
            score += 0.25

        if intent.wear_type and row["wear_type"] == intent.wear_type:
            score += 0.4

        if intent.style:
            text = str(row["search_text"]).lower()
            if intent.style in text:
                score += 0.25
            if intent.style == "formal" and row["occasion"] == "office":
                score += 0.35
            if intent.style == "ethnic" and row["wear_type"] == "ethnic":
                score += 0.35
            if intent.style == "sporty" and row["occasion"] == "sports":
                score += 0.35
        return score

    def _nearby_occasions(self, occasion: str) -> set[str]:
        return {
            "office": {"casual", "party"},
            "party": {"casual", "office"},
            "wedding": {"festive", "party"},
            "festive": {"wedding"},
            "vacation": {"casual"},
            "casual": {"vacation", "party"},
            "sports": {"casual"},
            "winter": {"casual"},
        }.get(occasion, set())

    def _graph_score(self, candidate_id: str, selected_ids: Iterable[str]) -> float:
        return sum(self.compatibility_graph[selected_id].get(candidate_id, 0.0) for selected_id in selected_ids)

    def _average_graph_strength(self, selected_ids: list[str]) -> float:
        if len(selected_ids) < 2:
            return 0.0
        scores = []
        for left in selected_ids:
            for right in selected_ids:
                if left != right:
                    scores.append(self.compatibility_graph[left].get(right, 0.0))
        return float(np.mean(scores)) if scores else 0.0

    def _quality_score(self, row: pd.Series) -> float:
        rating = row.get("rating", "")
        try:
            rating_value = float(rating)
        except (TypeError, ValueError):
            rating_value = 3.8
        return max(0.0, min(rating_value, 5.0)) / 20.0

    def _has_minimum_complete_outfit(self, items: list[RecommendedItem]) -> bool:
        roles = {item.role for item in items}
        has_main_or_top = bool(roles & {"main", "topwear"})
        has_bottom_or_main = "bottomwear" in roles or "main" in roles
        has_footwear = "footwear" in roles
        return has_main_or_top and has_bottom_or_main and has_footwear

    def _row_to_item(self, row: pd.Series) -> RecommendedItem:
        rating = row.get("rating", "")
        try:
            rating_value = float(rating)
        except (TypeError, ValueError):
            rating_value = None

        price = row.get("price_inr", 0)
        try:
            price_value = float(price)
        except (TypeError, ValueError):
            price_value = 0.0

        return RecommendedItem(
            id=str(row["id"]),
            name=str(row["name"]),
            brand=str(row["brand"]),
            role=str(row["role"]),
            category=str(row["category_label"]),
            occasion=str(row["occasion"]),
            color=str(row["color"]),
            price_inr=price_value,
            rating=rating_value,
            image_path=str(row["image_path"]),
            product_url=str(row.get("product_url", "")),
        )

    def _make_title(self, items: list[RecommendedItem], intent: UserIntent) -> str:
        main_item = next((item for item in items if item.role in {"main", "topwear"}), items[0])
        occasion = intent.occasion or main_item.occasion or "outfit"
        return f"{occasion.title()} outfit around {main_item.name}"

    def _build_reason(
        self,
        items: list[RecommendedItem],
        intent: UserIntent,
        graph_bonus: float,
    ) -> str:
        names = ", ".join(item.name for item in items)
        colors = ", ".join(dict.fromkeys(item.color for item in items if item.color))
        occasion = intent.occasion or "the requested occasion"
        gender_text = f" for {intent.gender}" if intent.gender else ""

        graph_line = (
            " Some pieces also appeared together in curated outfits, so the graph gives them a stronger compatibility score."
            if graph_bonus > 0
            else " The combination is selected by matching roles, occasion metadata, and color harmony."
        )

        return (
            f"This recommendation fits {occasion}{gender_text} because it combines the needed outfit roles: {names}. "
            f"The palette ({colors}) keeps the outfit visually coordinated while still giving contrast. "
            f"{graph_line}"
        )

    def _items_summary(self, items: list[RecommendedItem]) -> str:
        return "; ".join(
            f"{item.role_label}: {item.name} ({item.color}, {item.occasion})" for item in items
        )
