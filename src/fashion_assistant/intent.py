"""Small natural-language parser for the chat interface.

This is intentionally transparent.  A production version could use an LLM or a
trained classifier, but this rule-based parser is easy to debug and explain.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class UserIntent:
    message: str
    gender: str | None = None
    age: int | None = None
    occasion: str | None = None
    style: str | None = None
    wear_type: str | None = None
    budget_inr: int | None = None

    def search_query(self) -> str:
        parts = [self.message]
        for value in [self.gender, self.occasion, self.style, self.wear_type]:
            if value:
                parts.append(value)
        if self.age:
            parts.append(f"{self.age} years old")
        return " ".join(parts)


OCCASION_KEYWORDS = {
    "office": [
        "office",
        "work",
        "business",
        "meeting",
        "interview",
        "corporate",
        "formal",
        "presentation",
    ],
    "party": ["party", "date", "dinner", "night out", "cocktail", "club", "evening"],
    "wedding": ["wedding", "marriage", "reception", "ceremony"],
    "festive": ["festive", "festival", "diwali", "eid", "pooja", "celebration"],
    "casual": ["casual", "college", "daily", "weekend", "smart casual", "hangout"],
    "vacation": ["vacation", "holiday", "beach", "travel", "summer", "resort"],
    "sports": ["gym", "sports", "workout", "running", "athleisure", "training"],
    "winter": ["winter", "cold", "jacket", "coat", "sweater"],
}

STYLE_KEYWORDS = {
    "formal": ["formal", "professional", "business", "interview", "corporate"],
    "smart casual": ["smart casual", "dinner date", "polished casual"],
    "casual": ["casual", "relaxed", "daily", "weekend", "street"],
    "party": ["party", "stylish", "night out", "glam", "cocktail"],
    "ethnic": ["ethnic", "traditional", "festive", "saree", "kurta", "sherwani"],
    "sporty": ["sporty", "gym", "workout", "athleisure"],
}


def parse_user_message(message: str) -> UserIntent:
    lowered = message.lower()
    intent = UserIntent(message=message)

    intent.gender = _detect_gender(lowered)
    intent.age = _detect_age(lowered)
    intent.occasion = _detect_from_keywords(lowered, OCCASION_KEYWORDS)
    intent.style = _detect_from_keywords(lowered, STYLE_KEYWORDS)
    intent.wear_type = _detect_wear_type(lowered, intent)
    intent.budget_inr = _detect_budget(lowered)
    return intent


def merge_profile(intent: UserIntent, profile: dict | None) -> UserIntent:
    """Let explicit sidebar/profile values override what the parser inferred."""
    if not profile:
        return intent

    for field in ["gender", "age", "occasion", "style", "wear_type", "budget_inr"]:
        value = profile.get(field)
        if value not in [None, "", "any"]:
            setattr(intent, field, value)
    return intent


def _detect_gender(text: str) -> str | None:
    if re.search(r"\b(woman|women|female|girl|lady)\b", text):
        return "women"
    if re.search(r"\b(man|men|male|boy|gentleman)\b", text):
        return "men"
    return None


def _detect_age(text: str) -> int | None:
    match = re.search(r"\b(\d{2})\s*(?:year|yr|y/o|yo|years old)?\b", text)
    if not match:
        return None
    age = int(match.group(1))
    if 12 <= age <= 90:
        return age
    return None


def _detect_budget(text: str) -> int | None:
    match = re.search(r"(?:under|below|less than|budget|within)\s*(?:rs\.?|inr)?\s*(\d{3,6})", text)
    if match:
        return int(match.group(1))
    return None


def _detect_from_keywords(text: str, mapping: dict[str, list[str]]) -> str | None:
    for label, keywords in mapping.items():
        if any(keyword in text for keyword in keywords):
            return label
    return None


def _detect_wear_type(text: str, intent: UserIntent) -> str | None:
    if intent.style == "ethnic" or any(word in text for word in ["ethnic", "traditional"]):
        return "ethnic"
    if any(word in text for word in ["western", "shirt", "jeans", "dress", "suit"]):
        return "western"
    return None
