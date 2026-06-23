# Architecture

## High-Level Diagram

```mermaid
flowchart TD
    A["Streamlit UI"] --> B["Chat request or product search"]
    B --> C["Intent parser"]
    C --> D["User profile merger"]
    E["products.csv"] --> F["Data loader"]
    G["outfits.csv"] --> F
    H["Product images"] --> I["Dominant color extraction"]
    F --> J["TF-IDF product matrix"]
    F --> K["Curated outfit TF-IDF matrix"]
    F --> L["Compatibility graph"]
    I --> M["Color feature"]
    D --> N["Recommendation ranker"]
    J --> N
    K --> N
    L --> N
    M --> N
    N --> O["Complete outfit slots"]
    O --> P["Explainability layer"]
    P --> Q["Optional Gemini rewrite"]
    Q --> A
```

## Components

### 1. Data Layer

Files:

- `products.csv`
- `outfits.csv`
- `images/`

Code:

- `src/fashion_assistant/data_loader.py`
- `src/fashion_assistant/features.py`

The loader adds:

- `role`: topwear, bottomwear, footwear, accessory, layer, or main.
- `search_text`: combined product metadata used for TF-IDF.
- `image_path`: absolute image path for the UI.
- `color`: broad color feature from metadata or image pixels.

### 2. Intent Layer

File:

- `src/fashion_assistant/intent.py`

It detects:

- gender
- age
- occasion
- style
- wear type
- budget

This keeps the prototype understandable without depending fully on prompt engineering.

### 3. Recommendation Layer

File:

- `src/fashion_assistant/recommender.py`

The score combines:

- TF-IDF text similarity
- exact and nearby occasion match
- gender and wear type match
- curated outfit compatibility graph strength
- color harmony
- product rating
- budget penalty

### 4. Explainability Layer

The default explanation is rule-based and always available. Gemini can optionally rewrite it if an API key is set.

File:

- `src/fashion_assistant/llm.py`

### 5. Interface Layer

File:

- `app.py`

The app has:

- Chat Assistant
- Compatibility Engine
- Dataset Analysis
