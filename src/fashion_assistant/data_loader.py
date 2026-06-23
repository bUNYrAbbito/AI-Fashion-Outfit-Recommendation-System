"""Load and prepare the Dare XAI fashion dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .features import classify_role, estimate_image_color, first_color_from_text, make_search_text


@dataclass
class FashionData:
    """Container for the prepared dataset."""

    root_dir: Path
    products: pd.DataFrame
    outfits: pd.DataFrame


def load_fashion_data(root_dir: str | Path | None = None) -> FashionData:
    """Load products and outfits, then add helper columns used by the app."""
    root = Path(root_dir or Path.cwd()).resolve()
    products_path = root / "products.csv"
    outfits_path = root / "outfits.csv"

    if not products_path.exists() or not outfits_path.exists():
        raise FileNotFoundError(
            "products.csv and outfits.csv must be present in the project root."
        )

    products = pd.read_csv(products_path)
    outfits = pd.read_csv(outfits_path)

    products = products.fillna("")
    outfits = outfits.fillna("")

    products["role"] = products.apply(
        lambda row: classify_role(row["category"], row.get("wear_type", "")), axis=1
    )
    products["search_text"] = products.apply(lambda row: make_search_text(row.to_dict()), axis=1)
    products["image_path"] = products["image"].apply(lambda value: str((root / value).resolve()))
    products["color"] = products.apply(
        lambda row: _product_color(row["search_text"], row["image_path"]), axis=1
    )

    outfits["outfit_text"] = outfits.apply(_make_outfit_text, axis=1)

    return FashionData(root_dir=root, products=products, outfits=outfits)


def _product_color(search_text: str, image_path: str) -> str:
    text_color = first_color_from_text(search_text)
    if text_color:
        return text_color
    return estimate_image_color(image_path)


def _make_outfit_text(row: pd.Series) -> str:
    fields = [
        "gender",
        "wear_type",
        "occasion",
        "theme",
        "hero",
        "second",
        "layer",
        "footwear",
        "accessory_1",
        "accessory_2",
        "palette",
        "stylist_rationale",
    ]
    return " ".join(str(row.get(field, "") or "") for field in fields)
