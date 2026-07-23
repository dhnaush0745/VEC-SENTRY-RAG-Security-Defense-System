# 🛡️ VEC-SENTRY
 
**Dynamic Geometric Profiling for Detecting Embedding-Space Hijacking in RAG Systems**
 
VEC-SENTRY is a runtime defense layer for Retrieval-Augmented Generation (RAG) pipelines. It sits between the vector-store retriever and the LLM, auditing every retrieved chunk with three independent geometric/structural signals before it is allowed into the context window. The notebook builds the attacks, builds the defense, evaluates it end‑to‑end on a public QA corpus, and ships a live Gradio app so any document (or the benchmark corpus) can be queried and attacked interactively.
 
- **Research area:** Vector & embedding weaknesses in RAG security
- **Hardware target:** Google Colab Free Tier (T4 GPU)
- **Stack:** 100% free & open-source (Sentence-Transformers, FAISS, spaCy, NetworkX, Gradio)
---
 
## Why this exists
 
RAG systems trust whatever the retriever hands back. A poisoned document that scores high cosine similarity to a query gets read by the LLM as ground truth — even if it was never a legitimate part of the corpus. VEC-SENTRY assumes the attacker can insert arbitrary vectors into the index and asks: **can we tell a legitimate top‑K neighbor from an adversarial one using only geometry and corpus structure, without a content classifier?**
 
## The threat model — three poisoning attacks
 
| Attack | Idea | What it exploits |
|---|---|---|
| **A — Bias Injection** | Factually-correct text with manipulated framing, plus token-level perturbation to push cosine similarity toward the target query | Content filters check facts, not framing |
| **B — PGD Adversarial Embedding** | Projected Gradient Descent directly in embedding space: start from a real chunk, optimize the vector inside an ε-ball to maximize similarity to the query, inject the resulting embedding straight into FAISS | Cross-model transferability of adversarial vectors |
| **C — One-Shot Orphan Chunk** | A standalone chunk with no structural lineage (no neighboring chunks from a real source document) that still dominates top‑K for a target query | Blind spot of similarity-only retrieval — no document ever validates chunk provenance |
 
Each attack is injected into its own poisoned FAISS index (plus a clean baseline index) so every scenario can be evaluated independently.
 
## The defense — three pillars
 
VEC-SENTRY audits every retrieved chunk with three signals, each targeting a different attack:
 
**Pillar 1 — Semantic Cohort Analysis (SCA)**
For the K chunks in a retrieval result, compute their centroid and score each chunk by its cosine distance from that centroid, normalized within the cohort's spread. A chunk over‑optimized for the query but geometrically detached from the rest of the topic cluster is flagged as an outlier.
 
**Pillar 2 — Cross-Model Manifold Validation (CMV)**
Re-rank the same query against a second, structurally different embedding model (shadow model). A legitimate chunk ranks similarly in both models; an adversarial vector, over-fit to the primary model's manifold, drops sharply in the shadow ranking. Scored as Rank Divergence Score (RDS) = |primary_rank − shadow_rank| / search_depth.
 
**Pillar 3 — Contextual Fragmentation Re-Assembly (CFR)**
Checks whether a chunk has real "siblings" — adjacent chunks from the same source document in the corpus. Orphan chunks (negative/missing source ID, or no neighbors at the expected chunk position) score high on the Lineage Gap Score (LGS) = 1 − average cosine similarity to adjacent chunks.
 
**Gated risk decision**
Rather than a single weighted threshold, the engine uses a two-stage gate:
1. **CFR hard gate** — if `cfr_score > 0.65`, flag immediately (CFR alone is the most reliable signal for orphan-chunk attacks).
2. **SCA ∧ CMV consensus gate** — otherwise, flag only if `sca_score > 0.55` **and** `cmv_score > 0.50` simultaneously, which suppresses single-pillar false positives.
A composite `risk_score = 0.10·SCA + 0.25·CMV + 0.65·CFR` is still computed and logged for visualization, but it is not what drives the flag decision.
 
**Defense modes** applied to flagged chunks:
- `hard_block` — remove flagged chunks from context entirely
- `soft_warn` — keep all chunks, prepend a `[RISK: x.xx]` tag
- `re_rank` — push flagged chunks to the bottom of the context list
## Notebook structure
 
| Phase | What it does |
|---|---|
| 0 | Environment setup — installs dependencies, forces a runtime restart for FAISS compatibility, sets global config (`CFG`), seeds, device detection |
| 1 | Loads NQ-Open (Natural Questions), chunks it with metadata (source doc id, chunk position — required by CFR), loads primary (`all-mpnet-base-v2`) and shadow (`all-MiniLM-L6-v2`) encoders, builds FAISS indexes for both |
| 2 | Implements the three attacks and injects them into three separate poisoned indexes + one clean baseline |
| 3 | Baseline (undefended) top‑K retrieval — establishes the raw Attack Success Rate |
| 4 | Implements the three pillars and the gated risk-scoring engine |
| 5 | Full evaluation loop across all attack types — Detection Rate, FPR, ASR reduction, F1, ROC-AUC |
| 6 | Ablation study across all 7 pillar combinations (each pillar alone, each pair, all three) |
| 7 | Paper-ready figures: ablation heatmap, ROC curves, before/after ASR, UMAP embedding-space plot, latency overhead, risk-score distributions, defense-mode comparison, threshold sensitivity sweep |
| 8 | Interactive `ipywidgets` demo — query the benchmark corpus under any attack scenario and inspect the audit output live |
| 9 | User document upload (PDF/DOCX/TXT) → builds a FAISS index from *your* document, extracts a spaCy-based knowledge graph (entities + relations), generates **document-aware attacks** targeting the document's own high-degree entities, and visualizes the attack path on the KG |
| 10 | **VEC-SENTRY Live System** — a single-cell Gradio app that reloads all saved state and serves everything above (corpus queries, document upload, pillar/threshold toggles, evaluation charts) as one live web UI |
 
> Note: the notebook contains several superseded/duplicate "Cell 10" attempts (iterating on the Gradio UI) — the last code cell before the final download cell is the one that should actually be run.
 
## Setup & run order
 
Designed for Google Colab (free T4 tier). Recommended order:
 
1. **Cell 0.1** — installs all core dependencies (`sentence-transformers`, `faiss-cpu`, `chromadb`, `datasets`, `transformers`, `accelerate`, `bitsandbytes`, `langchain`, `scikit-learn`, `umap-learn`, `ragas`, plus Phase 9 extras: `spacy`, `pyvis`, `networkx`, `pdfminer.six`, `python-docx`)
2. **Runtime restart cell** — required because `faiss-cpu` needs a fresh process after install (Colab: click Cancel if prompted, the cell force-kills the process itself)
3. **Cell 0.2** — imports, seeds, device detection, and global `CFG`
4. Run Phases 1 → 4 in order to build indexes and define the defense
5. Run Phase 5 (evaluation) and Phase 6 (ablation) for results
6. Run Phase 7 for figures
7. **Cell 10.1** — saves all state to `results/state/` (FAISS indexes, embeddings, metadata, config) so the live system doesn't need to re-run encoding
8. **Final Cell 10** — installs Gradio + Phase 9 deps and launches the live app
9. **Final cell** — zips everything under `results/` for download (figures, CSV/JSON data tables, KG HTML files)
## Key configuration (`CFG`)
 
| Parameter | Value | Meaning |
|---|---|---|
| `top_k` | 5 | Chunks retrieved per query |
| `chunk_size` / `chunk_overlap` | 256 / 50 | Word-level chunking |
| `n_corpus_docs` | 2000 | Corpus size (scalable) |
| `n_test_queries` | 200 | Queries evaluated per attack type |
| `n_attack_docs` | 100 | Poisoned chunks injected per attack type |
| `pgd_steps` / `pgd_step_size` / `pgd_epsilon` | 30 / 0.01 / 0.15 | PGD attack (Attack B) hyperparameters |
| `cfr_gate` | 0.65 | Hard-block threshold for Pillar 3 |
| `sca_gate` / `cmv_gate` | 0.55 / 0.50 | Consensus thresholds for Pillars 1 & 2 |
| `primary_model` / `shadow_model` | `all-mpnet-base-v2` / `all-MiniLM-L6-v2` | Encoder pair for CMV |
 
## Outputs
 
Running the full pipeline produces:
- `results/figures/*.pdf` — print-quality (300 DPI) figures for the paper
- `results/data/*.csv`, `*.json` — evaluation tables, ablation results, per-query audit logs
- `results/state/*` — saved indexes/embeddings/metadata for the live system
- KG HTML files (interactive pyvis knowledge graphs) from Phase 9
## Tech stack
 
- **Embeddings:** Sentence-Transformers (`all-mpnet-base-v2` primary, `all-MiniLM-L6-v2` shadow)
- **Vector index:** FAISS
- **NLP / KG:** spaCy (`en_core_web_sm`), NetworkX, pyvis
- **Document parsing:** pdfminer.six, python-docx
- **Evaluation:** scikit-learn (ROC-AUC, F1, precision/recall), UMAP
- **UI:** Gradio (live system), ipywidgets (in-notebook demo)
## Limitations / things to know before citing this as a result
 
- Risk thresholds (`cfr_gate`, `sca_gate`, `cmv_gate`) were manually tuned on this specific corpus/model pair and are not validated as universal defaults.
- The composite `risk_score` weights are cosmetic/logging-only under the current gated architecture — the actual flagging logic is the two-stage gate, not the weighted sum.
- CMV depends on having a genuinely different shadow model architecture; the strength of the signal will vary with the primary/shadow pairing.
- Evaluated only on NQ-Open plus synthetically generated attacks — no adversarial adaptation loop (i.e., an attacker aware of VEC-SENTRY's gates is not modeled).
