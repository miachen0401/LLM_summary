# RAG Dense Retriever Training System
### Proteomics & Biological Pathway Domain

---

## Project Overview

This project implements a **Dense Passage Retrieval (DPR)** training pipeline for building a Retrieval-Augmented Generation (RAG) system specialized in proteomics data insights, biological pathway analysis, and payload understanding. The system fine-tunes a **BGE (BAAI General Embedding)** model to learn semantic mappings between natural language questions and their corresponding answers about protein structures, signaling pathways, and therapeutic payloads.

The trained retriever can be integrated into a RAG pipeline where:
1. User submits a natural language question
2. Retriever finds semantically similar passages from a knowledge base
3. Retrieved passages are fed to an LLM for answer generation

---

## Architecture

```
                    ┌─────────────────┐
                    │  User Question  │
                    └────────┬────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │       BGE Encoder           │
              │  (Fine-tuned on Q&A pairs)  │
              └─────────────┬───────────────┘
                            │
                            ▼
              ┌─────────────────────────────┐
              │     Question Embedding      │
              │         (768-dim)           │
              └─────────────┬───────────────┘
                            │
                            ▼
              ┌─────────────────────────────┐
              │   Cosine Similarity Search  │
              │   against Answer Embeddings │
              └─────────────┬───────────────┘
                            │
                            ▼
              ┌─────────────────────────────┐
              │   Top-K Retrieved Passages  │
              └─────────────────────────────┘
```

---

## Technical Details

### 1. Model: BGE (BAAI General Embedding)

| Parameter | Value |
|-----------|-------|
| Base model | `BAAI/bge-base-en-v1.5` |
| Embedding dimension | 768 |
| Max sequence length | 512 tokens |
| Pooling strategy | Mean pooling over valid tokens |

### 2. Training Objective: Contrastive Learning with In-Batch Negatives

**Loss Function:**
- Uses cross-entropy loss over similarity scores
- **Positive pairs:** `(question_i, answer_i)` from same example
- **Negative pairs:** `(question_i, answer_j)` where `i != j` (in-batch negatives)

**Mathematical formulation:**
```python
logits = Q @ A.T              # (B, B) similarity matrix
labels = [0, 1, 2, ..., B-1]  # diagonal is positive
loss = CrossEntropy(logits, labels)
```

This approach efficiently uses batch samples as negatives without requiring explicit hard negative mining.

### 3. Embedding Generation

- **Query prefix:** `"Represent this question for retrieving relevant passages: "`
- **L2 normalization** applied to all embeddings
- Enables direct cosine similarity via dot product

### 4. Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Batch size | 16 |
| Epochs | 50 (configurable) |
| Device | CUDA if available, else CPU |

---

## Data Format

The training data is stored in JSON format with the following structure:

```json
{
  "train": [
    {
      "question": "<natural language question>",
      "answer": "<corresponding passage/answer>"
    }
  ],
  "eval": [
    {
      "question": "<natural language question>",
      "answer": "<corresponding passage/answer>"
    }
  ]
}
```

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Q&A pairs | ~7,775 |
| File location | `data/proteomics_qa_dataset.json` |

### Question Types Covered

| Type | Example |
|------|---------|
| Protein function | `"What is the function of <Protein Name>?"` |
| Pathway queries | `"Which signaling pathway involves <Gene/Protein>?"` |
| Payload mechanism | `"How does <Payload> achieve cellular internalization?"` |
| Target identification | `"What are the known targets of <Drug/Antibody>?"` |
| Expression patterns | `"Where is <Protein> predominantly expressed?"` |

### Answer Characteristics

- Contains structured proteomics and biological data
- Includes protein IDs (UniProt), gene symbols, pathway annotations
- References biological databases (KEGG, Reactome, GO terms)
- Covers ADC payloads, linker chemistry, and conjugation mechanisms

---

## Code Structure

```
train_RAG.py
│
├── QADataset(Dataset)
│   ├── __init__(data_path)    # Load JSON, split train/eval
│   ├── __len__()              # Return dataset size
│   └── __getitem__(index)     # Return single Q&A pair
│
├── BGEEncoder(nn.Module)
│   ├── __init__(model_name)   # Load tokenizer + transformer
│   ├── mean_pooling()         # Pool token embeddings
│   └── encode(text, device)   # Generate normalized embeddings
│
├── DenseRetrievalLoss(nn.Module)
│   └── forward()              # Compute contrastive loss
│
└── train_bge_retriever()      # Main training loop
```

---

## Usage

### Training

```bash
python train_RAG.py
```

The script will:
1. Load Q&A pairs from `data/proteomics_qa_dataset.json`
2. Initialize BGE encoder from HuggingFace
3. Train for 50 epochs using contrastive loss
4. Print loss per epoch
5. Return trained model

### Inference (after training)

```python
# Encode all answers to build index
doc_embeddings = model.encode(all_answers, device).cpu().numpy()

# Encode query and find nearest neighbors
query_emb = model.encode([user_question], device).cpu().numpy()
similarities = query_emb @ doc_embeddings.T
top_k_indices = similarities.argsort()[-k:][::-1]
```

---

## Dependencies

```
torch
transformers
```

> Model weights (`BAAI/bge-base-en-v1.5`) are auto-downloaded from HuggingFace.

---

## Key Concepts

### Dense Retrieval vs Sparse Retrieval

| Approach | Method | Characteristics |
|----------|--------|-----------------|
| **Sparse** | BM25, TF-IDF | Keyword matching, exact term overlap |
| **Dense** | This project | Semantic similarity via learned embeddings |

### Why BGE?

- State-of-the-art performance on retrieval benchmarks
- Supports both symmetric and asymmetric search
- Efficient inference with normalized embeddings

### In-Batch Negatives

- **Training efficiency:** No separate negative sampling required
- **Scaling benefit:** Larger batches = more negatives = better gradient signal
- **Trade-off:** Requires sufficient GPU memory for larger batches
