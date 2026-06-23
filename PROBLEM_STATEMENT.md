# Dare XAI – Machine Learning & AI Engineer Intern Assignment
## Assignment: AI Fashion Outfit Recommendation System

### Timeline
* **Duration**: 2 Days
* **Submission Deadline**: 48 Hours from Assignment Receipt

---

### Background
At Dare XAI, we are currently building an AI-powered Fashion Assistant capable of recommending complete outfit combinations based on:
1. User preferences
2. Occasion and context
3. Outfit compatibility
4. Fashion metadata
5. Product images
6. User profile information
7. Conversational inputs

You are provided with a fashion dataset containing outfit images and metadata (`NEWDATASET` folder). Your task is to design and build an intelligent recommendation system that can understand fashion relationships and generate complete outfit suggestions.

---

### Problem Statement
A user should be able to interact with the system naturally.
* **Examples**:
  * *"I need an outfit for a business meeting."*
  * *"Suggest a smart casual outfit for a dinner date."*
  * *"I am attending a wedding next weekend."*
  * *"I am a 22-year-old male looking for a casual summer outfit."*

The system should recommend a complete outfit consisting of:
* Topwear
* Bottomwear
* Footwear
* Optional accessories / layering
along with the **reasoning/rationale** behind the recommendation.

---

### Core Requirements

#### 1. Dataset Analysis
Perform an analysis of the provided dataset (`NEWDATASET`) and explain:
* Dataset structure
* Available metadata
* Categories
* Potential challenges
* Data quality observations
* How the dataset can be used to generate outfit recommendations

#### 2. Outfit Compatibility Engine
Build a recommendation engine that can generate compatible outfit combinations.
* **Examples**:
  * **Input**: White Shirt  
    **Output**: Navy Chinos, Brown Loafers
  * **Input**: Black Oversized T-Shirt  
    **Output**: Grey Cargo Pants, White Sneakers
* The system should rank recommendations based on compatibility and relevance.

#### 3. User-Aware Recommendations
Incorporate user information such as:
* Gender
* Age
* Occasion
* Style preferences
* **Example User Profile**:
  * Male, 24 years old, Formal style, Interview occasion
  * The system should adapt recommendations accordingly.

#### 4. Conversational Fashion Assistant
Create a chat-based experience where users can ask for outfit recommendations naturally.
* **Example**: *"I need something stylish for a beach vacation."*
* The assistant should understand the request and generate relevant outfit suggestions.
* *Note: You may use Gemini API Free Tier, OpenAI, Claude, or any suitable LLM. Gemini Free Tier is sufficient for this assignment and is encouraged for rapid prototyping.*

#### 5. Explainability
Every recommendation should include reasoning.
* **Example**: *"Beige chinos pair well with a navy blazer because the combination maintains a professional appearance while providing visual contrast."*

---

### Expected Technical Approach
We encourage candidates to explore deeper Machine Learning concepts rather than relying solely on prompt engineering. Possible approaches include:

* **Computer Vision**:
  * CLIP, FashionCLIP, SigLIP
  * Vision Transformers
  * Image Embeddings
* **Recommendation Systems**:
  * Content-Based Recommendation
  * Similarity Search
  * Ranking Systems
  * Hybrid Recommendation Engines
* **Retrieval Systems**:
  * FAISS, Qdrant, Chroma
  * Vector Databases
* **LLM Integration**:
  * Gemini, OpenAI, Claude
* **Advanced Approaches (Bonus)**:
  * Multi-modal Retrieval
  * Hybrid Search
  * RAG Pipelines
  * Learning Compatibility Scores
  * Pairwise Ranking Models
  * Outfit Similarity Networks
  * Fashion Graph-Based Recommendations

---

### Expected Outcome
By the end of the assignment, we expect a working prototype capable of:
* Understanding user intent
* Retrieving relevant fashion items
* Matching compatible outfit combinations
* Using both image and metadata information
* Generating explainable recommendations
* Supporting conversational interaction

*Note: We are not looking for perfect accuracy. We are evaluating problem solving ability, technical depth, engineering approach, learning mindset, and system design thinking.*

---

### Deliverables

#### Mandatory
1. **Source Code**: Uploaded to GitHub.
2. **Architecture Diagram**: Explaining system components.
3. **Technical Documentation**: Detailed inside the repository.
4. **Working Prototype**: Running app/script.
5. **Video Demonstration (5–10 Minutes)**:
   * The video should cover: Dataset understanding, architecture, approach taken, key design decisions, system demo, challenges faced, and future improvements.

---

### Submission Instructions

#### GitHub Repository
1. Create a **Private GitHub Repository**.
2. Grant access to GitHub Username: **addygeek**
3. Include: Complete source code, README, setup instructions, and architecture explanation.

#### Demo Video
Upload the demonstration video to:
* YouTube (Unlisted) OR Google Drive (make link accessible)
* Share the link in the submission form.

---

### AI Usage Policy
You are free to use ChatGPT, Gemini, Claude, Cursor, GitHub Copilot, or any AI-assisted development tools.
However, you must fully understand and be able to explain:
* The architecture
* The code
* The ML approach
* The design decisions
We value learning ability and problem-solving skills more than prior expertise. Candidates who demonstrate strong understanding, curiosity, and ownership will be preferred.

---

### Evaluation Criteria

| Area | Weight |
|---|---|
| **ML / AI Depth** | 30% |
| **Recommendation Quality** | 20% |
| **Computer Vision Usage** | 15% |
| **Retrieval / Search Design** | 15% |
| **System Design** | 10% |
| **Documentation & Demo** | 10% |
| **Total** | **100%** |
