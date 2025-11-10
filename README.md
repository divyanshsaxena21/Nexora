🧥 Vibe Matcher — Nexora AI Prototype
🧠 Overview

At Nexora, AI bridges emotion and expression — turning how people feel into what they wear.
The Vibe Matcher prototype demonstrates how language embeddings can interpret subjective “vibes” (like energetic urban chic or cozy weekend comfort) and recommend fashion products that match those moods in real time.

🚀 Features

💬 Natural vibe input — users describe moods in free text

🧾 Semantic product embeddings — generated using all-mpnet-base-v2 (no API required)

🧭 Vector similarity search — cosine similarity ranks the top-3 best-fit items

⚡ Fast + local — runs 100% on-device via SentenceTransformers

📊 Evaluation metrics — latency tracking, similarity scoring, and visualization

🧩 Tech Stack
Component	Tool
Language Embeddings	SentenceTransformers
 (all-mpnet-base-v2)
Similarity Metric	Cosine similarity (sklearn)
Data Handling	Pandas + NumPy
Visualization	Matplotlib
Environment	Google Colab / Jupyter Notebook
🧰 How to Run

Open vibe_matcher.ipynb
 in Google Colab or Jupyter.

Run all cells — no API key required.

Enter a vibe query (e.g. "energetic urban chic") and view top-3 recommended items.

Review similarity scores and latency plot.

📊 Sample Output
Query	Top Match	Avg Similarity
energetic urban chic	Street Hoodie	0.55
cozy weekend comfort	Cozy Knit Sweater	0.60
beachy boho vibes	Beach Sandals	0.52
🔍 Reflection — Future Enhancements

Integrate FAISS or Pinecone for scalable vector retrieval

Add multimodal embeddings (CLIP) for visual vibe alignment

Fine-tune on fashion-specific corpora for nuance and tone

Enable user feedback loops (“more elegant”, “less sporty”)

Expand dataset and log retrieval metrics (precision@k, recall@k)

🏁 Summary

The Vibe Matcher prototype showcases how accessible AI can translate human “vibes” into meaningful fashion recommendations.
This experiment lays the foundation for Nexora’s future in emotion-aware retail discovery — blending creativity, data, and intuition.
