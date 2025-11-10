# Nexora — Vibe Matcher (AI Prototype)

A lightweight prototype that maps human "vibes" (free-text mood descriptions) to fashion product recommendations using sentence embeddings and vector similarity. Vibe Matcher demonstrates how language embeddings can translate subjective style prompts (e.g., "energetic urban chic", "cozy weekend comfort") into ranked product suggestions — all running locally with SentenceTransformers (no external APIs).

Status: Prototype / Experiment

---

## Key features

- 💬 Natural vibe input: describe a mood in plain text.
- 🧾 Semantic product embeddings: pre-computed using `all-mpnet-base-v2` (SentenceTransformers).
- 🧭 Vector similarity search: cosine similarity to rank top-N items.
- ⚡ Fast & local: runs on-device (CPU/GPU) — ideal for Colab or local Jupyter.
- 📊 Evaluation: latency logging, similarity scoring, and visualization (Matplotlib).
- 🔒 No API keys required.

---

## Tech stack

- Python
- sentence-transformers (all-mpnet-base-v2)
- scikit-learn (cosine similarity)
- pandas, numpy
- matplotlib
- Jupyter / Google Colab

---

## Quick start

Recommended: open `vibe_matcher.ipynb` in Google Colab or run it locally in Jupyter.

1. Clone the repo (optional if using Colab):
   ```bash
   git clone https://github.com/divyanshsaxena21/Nexora.git
   cd Nexora
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   If a `requirements.txt` is not present, at minimum install:
   ```bash
   pip install sentence-transformers scikit-learn pandas numpy matplotlib
   ```

3. Open the notebook:
   - In Colab: Upload/open `vibe_matcher.ipynb`
   - Locally: `jupyter notebook vibe_matcher.ipynb` or `jupyter lab`

4. Run all cells. Enter a vibe query in the provided input cell and view the top-3 recommended items, similarity scores, and a latency plot.

---

## Minimal usage example (script)

The notebook contains the full pipeline; here is a short example of the recommendation step:

```python
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load model
model = SentenceTransformer("all-mpnet-base-v2")

# Example data: product descriptions (or titles) and precomputed embeddings
products = pd.DataFrame({
    "id": [1,2,3],
    "title": ["Street Hoodie", "Cozy Knit Sweater", "Beach Sandals"],
    "description": [
        "Oversized hoodie, streetwear aesthetic",
        "Chunky knit sweater, warm and cozy",
        "Lightweight sandals, boho beach style"
    ],
})

# Compute embeddings for product corpus (do this once and cache)
product_embeddings = model.encode(products["description"].tolist(), convert_to_numpy=True)

# Query
query = "energetic urban chic"
query_emb = model.encode([query], convert_to_numpy=True)

# Similarity and top-k
sims = cosine_similarity(query_emb, product_embeddings)[0]
top_k_idx = np.argsort(-sims)[:3]

for rank, idx in enumerate(top_k_idx, start=1):
    print(f"{rank}. {products.iloc[idx]['title']} — similarity: {sims[idx]:.3f}")
```

---

## Sample output (illustrative)

| Query                  | Top Match          | Avg Similarity |
|------------------------|--------------------|----------------|
| energetic urban chic   | Street Hoodie      | 0.55           |
| cozy weekend comfort   | Cozy Knit Sweater  | 0.60           |
| beachy boho vibes      | Beach Sandals      | 0.52           |

---

## Evaluation & metrics

- Latency: measure model encode time and similarity search time per query.
- Ranking: report top-k similarity scores and qualitative assessment.
- Suggested metrics to log for larger evaluations: precision@k, recall@k, MRR.

---

## Future enhancements

- Integrate a vector DB (FAISS, Pinecone) for scalable retrieval.
- Add multimodal embeddings (CLIP) to align images and text.
- Fine-tune embeddings on fashion-specific corpora for nuance.
- Add user feedback loop for relevance tuning ("more elegant", "less sporty").
- Persist product embeddings and metadata; add offline indexing.

---

## Data & licensing

- This prototype uses a small demo dataset of product titles/descriptions (for demonstration only).
- License: please add a LICENSE file to declare the intended license for this project. If not specified, assume this repo is under the default GitHub terms until a license is provided.

---

## Contributing

PRs and issues are welcome. Consider:
- Adding a requirements.txt or environment.yml
- Adding a cached embeddings artifact and a script to (re)build embeddings
- Improving evaluation notebooks and visualization

---

## Acknowledgements

Built with SentenceTransformers and inspired by retrieval-based recommendation patterns.

---

## Contact

Maintainer: divyanshsaxena21 (repository owner)
