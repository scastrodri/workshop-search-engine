# Workshop: Build Your Own Search Engine

This repository contains my implementation of the
[Build Your Own Search Engine](https://aishippinglabs.com/workshops/2026-05-14-build-your-own-search-engine)
workshop by [Alexey Grigorev](https://github.com/alexeygrigorev) /
[AI Shipping Labs](https://aishippinglabs.com/).

The original workshop code is available at
https://github.com/alexeygrigorev/build-your-own-search-engine.

---

## Problem

Traditional search engines rely on exact keyword matching. While fast and computationally cheap, they suffer from a major flaw: they are blind to context and synonyms. If a user queries "I just signed up, is it too late to join?", a lexical engine will fail to retrieve documents that use words like "registered" or "enrolled" because the exact strings do not match. 

This project solves that gap by implementing and comparing multiple information retrieval techniques.

### What this builds

A search engine over FAQ documents, progressing from keyword matching to
semantic embeddings. The dataset is the DataTalks.Club Zoomcamp FAQ.

---

## Approaches covered

| Approach | File | Description | Pros | Cons |
|---|---|---|---|---|
| TF-IDF + Cosine Similarity (Lexical Search) | `search_engine.py` | Keyword-based search with field boosting and filters. Represents documents as sparse vectors based on word frequencies. | Extremely fast and lightweight. | Fails at capturing synonyms or semantic meaning. | 
| SVD / NMF Embeddings | `search-engine.ipynb` | Compresses the sparse TF-IDF matrices into dense vectors using Linear Algebra (Singular Value Decomposition and Non-Negative Matrix Factorization). | Attempts to group frequently co-occurring words into latent "topics". | Requires a massive corpus to learn language patterns effectively. On smaller datasets, it often performs worse than pure TF-IDF. |
| BERT Semantic Search | `semantic_search.py` | Uses a pre-trained Transformer model (`bert-base-uncased` via Hugging Face/PyTorch) to map sentences into a 768-dimensional semantic space. | Understands user intent, context, and synonyms right out of the box. Supports Symmetric Search (comparing queries against FAQs). | Computationally expensive; requires batching and ideally a GPU. |

---

## Key finding

Searching over the `question` field with BERT embeddings outperforms
TF-IDF on semantic queries — e.g. *"Is it too late to join?"* correctly
retrieves *"Can I still join after the start date?"* despite sharing no
keywords.

TF-IDF with field boosting (`question` weight 3x) remains competitive
for exact keyword matches and is significantly faster.

---

## Project structure

```
├── search_engine.py      # TextSearch class (TF-IDF)
├── semantic_search.py    # SemanticSearch class (BERT) — slow on CPU
├── utils.py              # make_batches helper
├── search-engine.ipynb   # Full exploration notebook
├── pyproject.toml        # Dependencies managed with uv
└── .gitignore
```
---

## Setup

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/scastrodri/workshop-search-engine.git
cd workshop-search-engine
uv sync
```

### Run TF-IDF search

```bash
uv run python search_engine.py
```

### Run BERT search

```bash
uv run python semantic_search.py
```

> ⚠️ BERT requires downloading `bert-base-uncased` (~400MB) on first run
> and is slow on CPU. For larger datasets, a GPU is recommended.

### Explore the full notebook

```bash
uv run jupyter notebook
```

Open `search-engine.ipynb` for the complete progression including
SVD, NMF, and BERT experiments.

---

## What I learned

- TF-IDF represents documents as sparse vectors weighted by term rarity.
  Field boosting improves precision by weighting the `question` field higher.
- SVD and NMF compress TF-IDF matrices into dense embeddings but remain
  bag-of-words — they miss synonyms and word order.
- BERT produces context-aware embeddings that capture semantic similarity.
  The trade-off is speed: 10+ minutes on CPU vs milliseconds for TF-IDF.
- Searching over the right field matters as much as the search algorithm.

---

## Next steps

This search engine is the foundation for RAG (Retrieval-Augmented Generation).
The natural next step is connecting it to a language model to generate answers
from retrieved documents — covered in the
[From RAG to Agents](https://aishippinglabs.com/workshops/2026-05-04-agentic-rag)
workshop.