# LatentSelf

> **Your thoughts never leave your machine. Your mind maps its own universe.**

A fully local "digital psychoanalysis" system that transforms your AI chat history into a 3D cognitive topology map. Using Qwen3-Embedding-8B, UMAP manifold learning, and HDBSCAN clustering, it reveals the latent structure of your thinking patterns — what you ask about, how your interests cluster, and how your cognitive focus evolves over time.

## What it does

1. **Ingests** your Google Gemini chat export (Google Takeout)
2. **Sessions** conversations by time proximity (15-min gap) + semantic similarity (embedding cosine)
3. **Embeds** each session into a 4096-dim vector using Qwen3-Embedding-8B
4. **Maps** the high-dimensional space to 3D via UMAP, clusters with HDBSCAN
5. **Interprets** clusters using statistical probing (centroid deviation analysis) + LLM-generated explanations
6. **Visualizes** everything as an interactive 3D "constellation map" in Streamlit

## Screenshot

> *Anthropic interpretability-inspired dark theme. Each point is a conversation session. Colors represent semantic clusters. Hover for details.*

## Architecture

```
data/raw/MyActivity.json
    │
    ▼  Phase 1: cleaner.py (time-based sessionization + noise filtering)
    │
    ▼  Phase 1.5: splitter.py (semantic sub-splitting via turn-level embedding similarity)
    │
data/processed/sessionized_data.json
    │
    ▼  Phase 2: embedder.py (Qwen3-Embedding-8B via vLLM, L2 normalized)
    │
data/processed/embeddings.npy
    │
    ▼  Phase 3: manifold.py (UMAP 4096→3D + HDBSCAN density clustering)
    │
    ▼  Phase 5: interpreter.py (centroid deviation probing + cluster naming)
    │
data/processed/interpretations.json
    │
    ▼  Phase 4: app.py (Streamlit + Plotly 3D visualization)
    │
http://localhost:8501
```

## Project Structure

```
LatentSelf/
├── data/
│   ├── raw/                          # Your Google Takeout data (gitignored)
│   │   └── MyActivity.json
│   └── processed/                    # Pipeline outputs (gitignored)
│       ├── sessionized_data.json
│       ├── embeddings.npy
│       └── interpretations.json
├── latentself/
│   ├── __init__.py
│   ├── pipeline/
│   │   ├── cleaner.py                # Phase 1: Time-based session splitting
│   │   └── splitter.py               # Phase 1.5: Semantic sub-splitting (GPU)
│   ├── engine/
│   │   ├── embedder.py               # Phase 2: Qwen3-Embedding-8B vectorization
│   │   ├── manifold.py               # Phase 3: UMAP + HDBSCAN
│   │   └── interpreter.py            # Phase 5: Cluster interpretability analysis
│   └── viz/
│       ├── __init__.py
│       └── app.py                    # Phase 4: Streamlit 3D visualization
├── requirements.txt
├── .gitignore
└── README.md
```

## Requirements

- Python 3.10+
- NVIDIA GPU with ~16GB VRAM (for Qwen3-Embedding-8B)
- [vLLM](https://github.com/vllm-project/vllm) v0.18+
- ~15GB disk for model weights (auto-downloaded)

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For the embedding steps (Phase 1.5 and Phase 2), you need vLLM in a separate environment:

```bash
pip install vllm transformers
```

### 2. Prepare your data

Export your Gemini chat history from [Google Takeout](https://takeout.google.com/), then place `MyActivity.json` into `data/raw/`:

```bash
mkdir -p data/raw
cp /path/to/your/MyActivity.json data/raw/
```

### 3. Run the pipeline

```bash
# Phase 1: Clean and sessionize
python -m latentself.pipeline.cleaner

# Phase 1.5: Semantic sub-splitting (requires GPU)
python -m latentself.pipeline.splitter

# Phase 2: Generate embeddings (requires GPU)
python -m latentself.engine.embedder

# Phase 3: UMAP + HDBSCAN
python -m latentself.engine.manifold

# Phase 5: Cluster interpretability analysis
python -m latentself.engine.interpreter
```

### 4. Launch visualization

```bash
streamlit run latentself/viz/app.py --server.address localhost
```

Open http://localhost:8501

## Visualization Features

- **3D Constellation Map**: UMAP-projected sessions as interactive 3D scatter plot
- **Cluster Detail Cards**: Each cluster has a name and deep semantic interpretation
- **Cluster Comparison**: Side-by-side analysis of distinguishing embedding dimensions
- **Dimension Heatmap**: Cluster x Dimension activation matrix (Z-Score)
- **Focus Mode**: Select a cluster to highlight it, dimming all others
- **Time Filtering**: Sidebar slider to filter sessions by date range
- **Session Table**: Expandable list with cluster names and content previews

## Interpretability Approach

Instead of naively labeling clusters by topic (which doesn't need embeddings), we analyze **what the embedding space itself captures**:

1. **Statistical Probing**: For each cluster, compute centroid deviation from global mean in each of 4096 dimensions. Find dimensions with highest Z-score — these are the cluster's "signature features".
2. **Extreme Sample Extraction**: For each signature dimension, find sessions with highest/lowest activation — these reveal what semantic concept that dimension encodes.
3. **LLM Bridge**: Feed the mathematical evidence (extreme samples, contrastive pairs) to an LLM, asking it to explain *why the vector space groups these together* — not just what they're about.

This answers: **"What geometric structure in the latent space causes these sessions to cluster?"**

## TODO

- [ ] **Time-axis animation**: Animate the constellation over time to show how cognitive focus evolves (which clusters grow/shrink, when new topics emerge)
- [ ] **Click-to-expand dialogue**: Click a point to see the full conversation (user prompts + AI responses) in a side panel
- [ ] **Support for other chat exports**: ChatGPT, Claude, etc.
- [ ] **Adaptive splitting threshold**: Auto-tune the semantic similarity threshold based on data distribution

## Tech Stack

| Component | Tool |
|-----------|------|
| Embedding Model | [Qwen/Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) |
| Model Inference | vLLM v0.18 |
| Dimensionality Reduction | UMAP (cosine metric, 3D projection) |
| Clustering | HDBSCAN (density-based, min_cluster_size=10) |
| Visualization | Streamlit + Plotly |
| Interpretability | Statistical probing + LLM-generated explanations |

## Privacy

All processing happens locally. Your conversation data never leaves your machine. The `data/` directory is gitignored — no personal data is committed to this repository.

## License

MIT
