# Technical Documentation

## Objective

Build a prototype that understands natural language fashion requests and recommends complete outfits containing topwear or mainwear, bottomwear when needed, footwear, and optional accessories.

## Data Preparation

`load_fashion_data()` reads `products.csv` and `outfits.csv`, fills missing values, and creates helper columns.

Product feature columns:

- `role`: fashion role derived from category.
- `search_text`: merged metadata used for retrieval.
- `image_path`: absolute local path for rendering product images.
- `color`: broad color feature from text or image.

Outfit feature columns:

- `outfit_text`: merged theme, occasion, palette, item names, and stylist rationale.

## Intent Parsing

The parser in `intent.py` extracts structured context from text:

- gender: men or women
- age: integer
- occasion: office, party, wedding, festive, casual, vacation, sports, winter
- style: formal, smart casual, casual, party, ethnic, sporty
- wear type: western or ethnic
- budget: optional INR value

The UI profile can override parsed values.

## Retrieval

The system uses `TfidfVectorizer` from scikit-learn.

Product documents are created from:

- name
- brand
- gender
- wear type
- category
- occasion
- tags
- description

The user query is converted into the same TF-IDF space, and cosine similarity retrieves relevant products.

## Compatibility Graph

Each curated outfit is treated as a set of compatible products. If two items appear in the same outfit, an edge is added between them.

During recommendation, candidate products receive a higher score when they are connected to already selected items.

## Image Feature

The assignment asks for use of product images. This prototype uses a lightweight computer vision feature:

1. Open the product image with Pillow.
2. Resize it for speed.
3. Remove near-white background pixels.
4. Average the remaining pixels.
5. Map the average RGB value to a broad color name.

This is not as advanced as CLIP or FashionCLIP, but it is easy to understand and still uses visual data.

## Ranking Formula

The final score is a weighted combination of:

- TF-IDF similarity
- gender match
- occasion match
- wear type match
- style match
- compatibility graph strength
- color harmony
- rating score
- budget penalty

This makes the system more than prompt engineering while keeping the implementation explainable.

## Explainability

Every recommendation includes:

- why it fits the occasion
- which outfit roles were selected
- how colors work together
- whether curated outfit compatibility influenced the score

If Gemini is enabled, the same facts are passed to Gemini only for wording improvement.

## Future Improvements

- Replace simple image color extraction with CLIP, FashionCLIP, or SigLIP embeddings.
- Store embeddings in FAISS or Chroma for faster retrieval on larger catalogs.
- Train a pairwise compatibility model from more curated outfits.
- Add body type, climate, size, and budget-aware personalization.
- Add feedback buttons and learn from accepted or rejected recommendations.
- Add a better LLM intent parser with JSON output validation.
