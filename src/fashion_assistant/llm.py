"""Optional Gemini integration for nicer final explanations.

The recommender works without an API key.  If a user sets GOOGLE_API_KEY or
GEMINI_API_KEY, the app can ask Gemini to rewrite the explanation in a more
natural assistant voice.
"""

from __future__ import annotations

import os


def improve_explanation_with_gemini(user_request: str, outfit_summary: str, fallback: str) -> str:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return fallback

    try:
        import google.generativeai as genai
    except ImportError:
        return fallback

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""
You are a concise fashion stylist. Rewrite the recommendation reason in 3-5
sentences. Do not invent products. Keep it practical and explain compatibility.

User request: {user_request}
Outfit: {outfit_summary}
Draft reason: {fallback}
"""
        response = model.generate_content(prompt)
        text = getattr(response, "text", "").strip()
        return text or fallback
    except Exception:
        return fallback
