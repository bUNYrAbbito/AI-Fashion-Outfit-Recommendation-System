from __future__ import annotations

import base64
import html
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fashion_assistant import FashionRecommender, load_fashion_data, parse_user_message  # noqa: E402


st.set_page_config(
    page_title="Dare XAI Fashion Assistant",
    layout="wide",
)


@st.cache_resource
def get_recommender() -> FashionRecommender:
    data = load_fashion_data(PROJECT_ROOT)
    return FashionRecommender(data)


def money(value: float) -> str:
    if not value:
        return "Price not listed"
    return f"Rs. {value:,.0f}"


def is_avif_file(path: str) -> bool:
    try:
        return Path(path).read_bytes()[:12].endswith(b"ftypavif")
    except OSError:
        return False


def display_product_image(image_path: str, alt_text: str) -> None:
    path = Path(image_path)
    if not path.exists():
        st.warning("Image missing")
        return

    if is_avif_file(str(path)):
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        safe_alt = html.escape(alt_text)
        st.markdown(
            f"""
            <img
                src="data:image/avif;base64,{encoded}"
                alt="{safe_alt}"
                style="width: 100%; max-height: 260px; object-fit: contain; border-radius: 6px;"
            />
            """,
            unsafe_allow_html=True,
        )
        return

    try:
        st.image(str(path), use_column_width=True)
    except Exception:
        st.warning("Image preview unavailable")


def sidebar_profile() -> dict:
    st.sidebar.header("User Profile")
    gender = st.sidebar.selectbox("Gender", ["any", "men", "women"], index=0)
    age = st.sidebar.number_input("Age", min_value=12, max_value=90, value=24)
    occasion = st.sidebar.selectbox(
        "Occasion",
        ["any", "office", "casual", "party", "wedding", "festive", "vacation", "sports", "winter"],
        index=0,
    )
    style = st.sidebar.selectbox(
        "Style",
        ["any", "formal", "smart casual", "casual", "party", "ethnic", "sporty"],
        index=0,
    )
    budget = st.sidebar.number_input("Budget INR", min_value=0, max_value=50000, value=0, step=500)
    use_llm = st.sidebar.toggle("Use Gemini explanation", value=False)

    return {
        "gender": None if gender == "any" else gender,
        "age": age,
        "occasion": None if occasion == "any" else occasion,
        "style": None if style == "any" else style,
        "budget_inr": None if budget == 0 else budget,
        "use_llm": use_llm,
    }


def display_item(item) -> None:
    display_product_image(item.image_path, item.name)
    st.markdown(f"**{item.role_label}**")
    st.markdown(item.name)
    st.caption(f"{item.brand} | {item.color} | {item.occasion}")
    st.caption(money(item.price_inr))
    if item.product_url:
        st.link_button("View product", item.product_url)


def display_recommendation(recommendation, number: int) -> None:
    st.subheader(f"{number}. {recommendation.title}")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Score", f"{recommendation.score:.2f}")
    metric_cols[1].metric("Items", len(recommendation.items))
    metric_cols[2].metric("Total", money(recommendation.total_price_inr))
    metric_cols[3].metric("Palette", recommendation.palette or "mixed")

    st.caption(recommendation.source)
    item_cols = st.columns(min(5, len(recommendation.items)))
    for col, item in zip(item_cols, recommendation.items):
        with col:
            display_item(item)
    st.info(recommendation.reason)
    st.divider()


def chat_page(recommender: FashionRecommender, profile: dict) -> None:
    st.header("Conversational Outfit Assistant")

    examples = [
        "I need an outfit for a business meeting.",
        "Suggest a smart casual outfit for a dinner date.",
        "I am attending a wedding next weekend.",
        "I am a 22-year-old male looking for a casual summer outfit.",
        "Style something around a white formal shirt.",
    ]
    selected_example = st.selectbox("Try a request", examples, key="example_request")
    if "query_text" not in st.session_state:
        st.session_state.query_text = selected_example
    if st.button("Use selected request"):
        st.session_state.query_text = selected_example

    query = st.text_input("Ask for an outfit", key="query_text")
    if not query.strip():
        st.warning("Enter an outfit request to start.")
        return

    with st.chat_message("user"):
        st.write(query)

    intent = parse_user_message(query)
    profile_for_engine = {key: value for key, value in profile.items() if key != "use_llm"}
    recommendations = recommender.recommend_from_chat(
        query,
        profile=profile_for_engine,
        top_k=3,
        use_llm=profile["use_llm"],
    )

    with st.chat_message("assistant"):
        if not recommendations:
            st.warning("No complete outfit could be built for this request.")
            return
        st.caption(
            f"Detected: gender={intent.gender or profile_for_engine.get('gender') or 'any'}, "
            f"occasion={intent.occasion or profile_for_engine.get('occasion') or 'any'}, "
            f"style={intent.style or profile_for_engine.get('style') or 'any'}"
        )
        for index, recommendation in enumerate(recommendations, start=1):
            display_recommendation(recommendation, index)


def product_page(recommender: FashionRecommender, profile: dict) -> None:
    st.header("Product Compatibility Engine")
    search_query = st.text_input("Product or style query", value="white formal shirt")
    gender = profile.get("gender")
    occasion = profile.get("occasion")

    results = recommender.search_products(search_query, gender=gender, occasion=occasion, top_k=10)
    view = results[["id", "name", "brand", "role", "category_label", "occasion", "color", "price_inr", "score"]].copy()
    view["score"] = view["score"].round(3)
    st.dataframe(view, use_container_width=True, hide_index=True)

    options = [f"{row['name']} ({row['id']})" for _, row in results.iterrows()]
    if not options:
        st.warning("No products matched the search.")
        return

    selected = st.selectbox("Build outfit around", options)
    product_id = selected.rsplit("(", 1)[-1].rstrip(")")
    product_row = recommender.products[recommender.products["id"].eq(product_id)].iloc[0]
    intent = parse_user_message(f"{search_query} {product_row['name']} {product_row['occasion']}")
    for key, value in profile.items():
        if key != "use_llm" and value not in [None, "", "any"]:
            setattr(intent, key, value)

    recommendation = recommender.recommend_around_product(
        product_id,
        intent=intent,
        use_llm=profile["use_llm"],
    )
    if recommendation:
        display_recommendation(recommendation, 1)
    else:
        st.warning("Could not build a complete outfit around this item.")


def dataset_page(recommender: FashionRecommender) -> None:
    st.header("Dataset Analysis")
    summary = recommender.dataset_summary()
    metric_cols = st.columns(4)
    metric_cols[0].metric("Products", summary["product_count"])
    metric_cols[1].metric("Curated outfits", summary["outfit_count"])
    metric_cols[2].metric("Genders", len(summary["gender_counts"]))
    metric_cols[3].metric("Roles", len(summary["role_counts"]))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Products by occasion")
        st.bar_chart(pd.Series(summary["occasion_counts"]))
    with col2:
        st.subheader("Products by role")
        st.bar_chart(pd.Series(summary["role_counts"]))

    st.subheader("Top product categories")
    st.dataframe(
        pd.DataFrame(
            [{"category": key, "count": value} for key, value in summary["category_counts"].items()]
        ),
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    recommender = get_recommender()
    st.title("Dare XAI Fashion Outfit Recommendation System")
    profile = sidebar_profile()

    chat_tab, product_tab, dataset_tab = st.tabs(
        ["Chat Assistant", "Compatibility Engine", "Dataset Analysis"]
    )
    with chat_tab:
        chat_page(recommender, profile)
    with product_tab:
        product_page(recommender, profile)
    with dataset_tab:
        dataset_page(recommender)


if __name__ == "__main__":
    main()
