# Dataset Analysis

## Dataset Size

- Products: 68
- Curated outfits: 25
- Product images: 68

## Product Gender Distribution

- women: 41
- men: 27

## Product Occasion Distribution

- casual: 15
- party: 13
- office: 12
- festive: 9
- wedding: 6
- sports: 5
- vacation: 4
- winter: 4

## Main Product Types

The dataset includes:

- Topwear: shirts, t-shirts, sweaters, sweatshirts, tops
- Bottomwear: jeans, trousers, chinos, shorts, skirts, leggings, track pants
- Footwear: sneakers, heels, loafers, formal shoes, sandals, boots, juttis
- Accessories: watches, clutches, handbags, caps, sunglasses, jewelry
- Main garments or sets: suits, sarees, dresses, kurta sets, sherwanis, sharara sets

## Observations

- The dataset is small but clean enough for a prototype.
- Curated outfits are valuable because they provide compatibility examples.
- Many women's outfits use one-piece items like dresses or sarees, so bottomwear is not always needed.
- Men's office and casual outfits have clearer topwear, bottomwear, footwear, and accessory structure.
- Metadata includes occasion and category, which are useful for retrieval and ranking.
- Product descriptions often include color words, which helps explain visual compatibility.

## Challenges

- Only 25 curated outfits are available, so a deep learning compatibility model would overfit.
- Some products have missing ratings or rating counts.
- Image backgrounds can affect simple color extraction.
- Natural language requests can be broad, so the parser uses keyword mapping and profile filters.

## Why This Approach Fits the Dataset

A hybrid system is more suitable than a large trained model here. The dataset is small, so the project uses:

- TF-IDF to retrieve products from metadata.
- Curated outfit co-occurrence to learn compatibility.
- Rule-based role completion to build full outfits.
- Simple image color features to include visual information.
