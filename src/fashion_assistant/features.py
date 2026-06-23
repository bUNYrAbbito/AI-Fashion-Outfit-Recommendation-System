"""Feature helpers used by the recommendation engine.

The assignment asks us to use both metadata and product images.  This file keeps
that logic small and easy to explain:

1. Metadata text is used for TF-IDF search.
2. Product category is mapped to a fashion role.
3. The image is used to estimate a broad color family when the text does not
   already mention a color.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


FOOTWEAR_CATEGORIES = {
    "boots",
    "ethnic-footwear",
    "flats",
    "formal-shoes",
    "heels",
    "loafers",
    "running-shoes",
    "sandals",
    "sneakers",
}

ACCESSORY_CATEGORIES = {
    "caps",
    "clutches",
    "earrings",
    "handbags",
    "necklaces",
    "sunglasses",
    "watches",
}

BOTTOM_CATEGORIES = {
    "chinos",
    "jeans",
    "leggings",
    "shorts",
    "skirts",
    "track-pants",
    "trousers",
}

LAYER_CATEGORIES = {
    "blazers",
    "denim-jackets",
    "long-coats",
    "nehru-jackets",
}

ONE_PIECE_OR_SET_CATEGORIES = {
    "activewear",
    "casual-dresses",
    "co-ord-sets",
    "kurta-sets",
    "maxi-dresses",
    "party-dresses",
    "salwar-suits",
    "sharara-sets",
    "sherwanis",
    "suits",
    "wedding-sarees",
}

TOP_CATEGORIES = {
    "casual-shirts",
    "formal-shirts",
    "linen-shirts",
    "party-shirts",
    "polo-tshirts",
    "sweaters",
    "sweatshirts",
    "tops",
    "tshirts",
}

COLOR_WORDS = [
    "black",
    "white",
    "grey",
    "gray",
    "navy",
    "blue",
    "green",
    "olive",
    "beige",
    "cream",
    "brown",
    "tan",
    "red",
    "maroon",
    "wine",
    "pink",
    "purple",
    "lavender",
    "magenta",
    "gold",
    "silver",
    "rust",
]

NORMALIZED_COLORS = {
    "gray": "grey",
    "wine": "maroon",
    "lavender": "purple",
    "magenta": "pink",
    "tan": "beige",
    "gold": "yellow",
    "silver": "grey",
    "rust": "brown",
}

RGB_COLOR_CENTERS = {
    "black": (25, 25, 25),
    "white": (245, 245, 245),
    "grey": (125, 125, 125),
    "navy": (20, 35, 90),
    "blue": (45, 95, 180),
    "green": (55, 130, 80),
    "olive": (105, 115, 50),
    "beige": (205, 185, 150),
    "cream": (235, 220, 185),
    "brown": (105, 70, 45),
    "red": (180, 45, 45),
    "maroon": (110, 30, 45),
    "pink": (210, 105, 140),
    "purple": (110, 70, 160),
    "yellow": (210, 170, 60),
}

NEUTRAL_COLORS = {"black", "white", "grey", "navy", "beige", "cream", "brown"}

GOOD_COLOR_PAIRS = {
    frozenset(("white", "navy")),
    frozenset(("white", "black")),
    frozenset(("white", "blue")),
    frozenset(("black", "grey")),
    frozenset(("black", "red")),
    frozenset(("navy", "brown")),
    frozenset(("navy", "green")),
    frozenset(("navy", "cream")),
    frozenset(("blue", "cream")),
    frozenset(("blue", "brown")),
    frozenset(("green", "brown")),
    frozenset(("olive", "cream")),
    frozenset(("beige", "brown")),
    frozenset(("cream", "brown")),
    frozenset(("red", "cream")),
    frozenset(("maroon", "grey")),
    frozenset(("purple", "cream")),
}


def classify_role(category: str, wear_type: str | None = None) -> str:
    """Convert dataset category into the role expected in a full outfit."""
    category = str(category or "").strip().lower()
    wear_type = str(wear_type or "").strip().lower()

    if category in FOOTWEAR_CATEGORIES or wear_type == "footwear":
        return "footwear"
    if category in ACCESSORY_CATEGORIES or wear_type == "accessory":
        return "accessory"
    if category in BOTTOM_CATEGORIES:
        return "bottomwear"
    if category in LAYER_CATEGORIES:
        return "layer"
    if category in ONE_PIECE_OR_SET_CATEGORIES:
        return "main"
    if category in TOP_CATEGORIES:
        return "topwear"
    return "main"


def make_search_text(row: dict) -> str:
    """Join important metadata fields into one document for TF-IDF retrieval."""
    fields = [
        "name",
        "brand",
        "gender",
        "wear_type",
        "category",
        "category_label",
        "occasion",
        "tags",
        "description",
    ]
    return " ".join(str(row.get(field, "") or "") for field in fields)


def first_color_from_text(text: str) -> str | None:
    """Find the first broad color word mentioned in metadata text."""
    text = f" {str(text or '').lower()} "
    for color in COLOR_WORDS:
        if f" {color} " in text or f"-{color}" in text or f"{color}-" in text:
            return NORMALIZED_COLORS.get(color, color)
    return None


@lru_cache(maxsize=256)
def estimate_image_color(image_path: str) -> str:
    """Estimate a product color from image pixels.

    Most product images have a white background.  We ignore very light pixels so
    the background does not overpower the garment color.
    """
    path = Path(image_path)
    if not path.exists():
        return "unknown"

    try:
        image = Image.open(path).convert("RGB")
    except OSError:
        return "unknown"

    image.thumbnail((120, 120))
    pixels = np.asarray(image).reshape(-1, 3)

    # Remove near-white backgrounds and very dark shadows.
    not_background = ~((pixels[:, 0] > 238) & (pixels[:, 1] > 238) & (pixels[:, 2] > 238))
    useful_pixels = pixels[not_background]
    if len(useful_pixels) < 40:
        useful_pixels = pixels

    avg = useful_pixels.mean(axis=0)
    return nearest_color_name(avg)


def nearest_color_name(rgb: Iterable[float]) -> str:
    """Map an RGB value to the closest color center."""
    rgb_array = np.array(list(rgb), dtype=float)
    best_color = "unknown"
    best_distance = float("inf")
    for color, center in RGB_COLOR_CENTERS.items():
        distance = np.linalg.norm(rgb_array - np.array(center, dtype=float))
        if distance < best_distance:
            best_color = color
            best_distance = distance
    return best_color


def harmony_score(color_a: str, color_b: str) -> float:
    """Simple color compatibility score between 0 and 1."""
    color_a = NORMALIZED_COLORS.get(str(color_a or "").lower(), str(color_a or "").lower())
    color_b = NORMALIZED_COLORS.get(str(color_b or "").lower(), str(color_b or "").lower())

    if "unknown" in {color_a, color_b} or not color_a or not color_b:
        return 0.45
    if color_a == color_b:
        return 0.75
    if color_a in NEUTRAL_COLORS or color_b in NEUTRAL_COLORS:
        return 0.85
    if frozenset((color_a, color_b)) in GOOD_COLOR_PAIRS:
        return 1.0
    return 0.35


def average_harmony(candidate_color: str, selected_colors: list[str]) -> float:
    """Average color harmony between a candidate and already selected items."""
    if not selected_colors:
        return 0.5
    scores = [harmony_score(candidate_color, color) for color in selected_colors]
    return float(np.mean(scores))
