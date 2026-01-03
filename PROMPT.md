# AIML Hackathon 2526 - Passage Ranking Challenge

## Overview

This project provides 4 passage ranking baselines for the MS MARCO dataset:

| # | Name | Model | Expected MRR@10 |
|---|------|-------|-----------------|
| 1 | Neural Re-Ranker | GloVe + BM25 + MLP | ~0.25-0.30 |
| 2 | BM25 + DFR + RF | Random Forest | ~0.25-0.30 |
| 3 | Cross-Encoder | sentence-transformers (limited) | ~0.28-0.35 |
| 4 | BERT CLS | BERT + 2-class classification | ~0.40-0.50 |

---

## Task Description

**Goal**: Given a query and a set of candidate passages, rank passages by relevance.

**Metric**: MRR@10 (Mean Reciprocal Rank at 10)

**Dataset**: MS MARCO v2 (sampled)
- Training: Up to 10,500 queries (~5 passages each)
- Validation (Public): 1,000 queries (~100 candidates each)
- Test Private: 1,000 queries (~100 candidates each)

---

## Eligibility

**IMPORTANT:**
- You must be a student of Sapienza University.
- You **MUST** join the Kaggle Competition using the link on Google Classroom.
- Sharing the link outside the course is **forbidden**.
- **Mandatory Environment:** You MUST use Kaggle Notebooks. Local execution or Colab is not supported for submission validation.

---

## Requirements

### Python Environment
```bash
conda create -n aiml_hackathon python=3.10
conda activate aiml_hackathon
pip install -r requirements.txt
```

### Key Dependencies (requirements.txt)
```
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.0
pytorch-lightning>=2.0.0
scikit-learn>=1.3.0
pandas>=2.0.0
rank_bm25>=0.2.2
tqdm>=4.65.0
numpy>=1.24.0
```

### External Resources (Auto-Downloaded)
1. **Dataset**: MS MARCO sampled (~44MB)
   - URL: `https://github.com/fabsilvestri/aiml_hackathon_data/releases/download/v1.0/msmarco_sampled.zip`
   - Auto-downloaded by `utils.load_data()`
   - **⚠️ IMPORTANT**: The ZIP contains files at root (no folder), so extract INTO `msmarco_sampled/`:
     ```python
     os.makedirs("msmarco_sampled", exist_ok=True)
     with zipfile.ZipFile("msmarco_sampled.zip", 'r') as zf:
         zf.extractall("msmarco_sampled")  # NOT "." !!
     ```
   
2. **GloVe Embeddings**: 840B 300D (~2GB)
   - URL: `http://nlp.stanford.edu/data/glove.840B.300d.zip`
   - Auto-downloaded by `utils.download_infersent_resources()`
   - Required for Baseline #1 and #2

3. **Pre-trained Models** (from HuggingFace, auto-cached):
   - `sentence-transformers/all-MiniLM-L6-v2` (Baseline #3)
   - `bert-base-uncased` (Baseline #4)

---

## File Structure

aiml_hackathon_2526/
├── starter_kit/                      # (Optional) Student submission templates
├── kaggle_setup/                     # (Notes) Kaggle Setup Scripts
├── data/                             # Dataset creation scripts
├── msmarco_sampled/                  # Dataset (auto-downloaded)
├── models/                           # Downloaded model files (GloVe, etc.)
├── outputs/                          # Results in TREC format
├── requirements.txt                  # Python dependencies
├── PROMPT.md                         # Project Context
├── README.md                         # Project overview
├── WALKTHROUGH.md                    # Instructor Guide
├── INSTRUCTIONS_FOR_STUDENTS.md      # Student Guide
├── INSTRUCTIONS.md                   # (Legacy) Rules
```

---

# BASELINE SPECIFICATIONS

Each baseline is defined precisely below. Implementations must match these specifications exactly.

---

## Baseline #1: Neural Re-Ranker (GloVe + BM25)

**Type**: PyTorch Module

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         NeuralReRanker                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   INPUTS:                                                           │
│   ├── q_ids: (B, seq_len) - tokenized query indices                 │
│   ├── q_len: (B,) - query lengths                                   │
│   ├── p_ids: (B, seq_len) - tokenized passage indices               │
│   ├── p_len: (B,) - passage lengths                                 │
│   └── bm25_score: (B,) - pre-computed BM25 scores                   │
│                                                                     │
│   EMBEDDING LAYER:                                                  │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  nn.Embedding.from_pretrained(glove_840B_300d, freeze=True) │   │
│   │  padding_idx=0                                              │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│   MEAN POOLING:                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  q_emb = mean_pool(q_ids, q_len)  →  (B, 300)               │   │
│   │  p_emb = mean_pool(p_ids, p_len)  →  (B, 300)               │   │
│   │                                                             │   │
│   │  Formula: sum(embeddings * mask) / length                   │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│   PROJECTION LAYERS:                                                │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  query_proj  = nn.Linear(300, 256)                          │   │
│   │  passage_proj = nn.Linear(300, 256)                         │   │
│   │                                                             │   │
│   │  q_proj = query_proj(q_emb)   →  (B, 256)                   │   │
│   │  p_proj = passage_proj(p_emb) →  (B, 256)                   │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│   FEATURE EXTRACTION (4 features):                                  │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  1. cosine_raw = cosine_sim(q_emb, p_emb)      # raw 300D   │   │
│   │  2. cosine_proj = cosine_sim(q_proj, p_proj)   # proj 256D  │   │
│   │  3. dot_proj = (q_proj · p_proj) / 256         # normalized │   │
│   │  4. bm25_norm = bm25 / (max(|bm25|) + 1e-8)    # normalized │   │
│   │                                                             │   │
│   │  features = stack([cosine_raw, cosine_proj, dot_proj,       │   │
│   │                    bm25_norm], dim=-1)  →  (B, 4)           │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│   SCORER MLP:                                                       │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  nn.Sequential(                                             │   │
│   │      nn.Linear(4, 32),                                      │   │
│   │      nn.ReLU(),                                             │   │
│   │      nn.Dropout(0.2),                                       │   │
│   │      nn.Linear(32, 1)                                       │   │
│   │  )                                                          │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│   OUTPUT: score (B,) - relevance score per query-passage pair       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Implementation Details

| Component | Specification |
|-----------|---------------|
| Embeddings | GloVe 840B 300D, **frozen** (`freeze=True`) |
| Embedding dim | 300 |
| Projection dim | 256 |
| Query projection | `nn.Linear(300, 256)` |
| Passage projection | `nn.Linear(300, 256)` |
| Number of features | 4 |
| Scorer hidden | 32 |
| Scorer dropout | 0.2 |
| Loss | `nn.MarginRankingLoss(margin=1.0)` |
| Optimizer | `AdamW(lr=1e-3, weight_decay=0.01)` |

### Training

```python
# Loss function
criterion = nn.MarginRankingLoss(margin=1.0)

# For each batch: (query, pos_passage, neg_passage)
pos_score = model(q, q_len, p_pos, p_pos_len, bm25_pos)
neg_score = model(q, q_len, p_neg, p_neg_len, bm25_neg)
loss = criterion(pos_score, neg_score, torch.ones_like(pos_score))
```

---

## Baseline #2: BM25 + DFR + Random Forest

**Type**: Scikit-Learn Pipeline

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    BM25 + DFR + Random Forest                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   INPUTS:                                                           │
│   ├── query: str                                                    │
│   └── passage: str                                                  │
│                                                                     │
│   TOKENIZATION:                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  def tokenize(text):                                        │   │
│   │      return text.lower().split()                            │   │
│   │                                                             │   │
│   │  query_tokens = tokenize(query)                             │   │
│   │  passage_tokens = tokenize(passage)                         │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│   FEATURE EXTRACTION (7 features):                                  │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  Feature 1: BM25 score                                      │   │
│   │      bm25 = BM25Okapi(tokenized_passages)                   │   │
│   │                                                             │   │
│   │  Feature 2: DFR score (Divergence From Randomness)          │   │
│   │      dfr = DFRScorer(tokenized_passages, model='In-L2')     │   │
│   │                                                             │   │
│   │  Feature 3: GloVe Cosine Similarity                         │   │
│   │      embedder = InferSentEmbedder(glove_path)               │   │
│   │                                                             │   │
│   │  Feature 4: TF-IDF cosine similarity                        │   │
│   │      vectorizer = TfidfVectorizer(max_features=10000)       │   │
│   │      tfidf_sim = cosine_similarity(q_vec, p_vec)[0]         │   │
│   │                                                             │   │
│   │  Feature 5: Query length (# tokens)                         │   │
│   │                                                             │   │
│   │  Feature 6: Passage length (# tokens)                       │   │
│   │                                                             │   │
│   │  Feature 7: Term overlap count                              │   │
│   │                                                             │   │
│   │  features = [bm25, dfr, glove, tfidf, q_len, p_len, overlap]│   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│   CLASSIFIER:                                                       │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  RandomForestClassifier(                                    │   │
│   │      n_estimators=100,                                      │   │
│   │      n_jobs=-1                                              │   │
│   │  )                                                          │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│   OUTPUT: P(relevant) = model.predict_proba(features)[:, 1]         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Implementation Details

| Component | Specification |
|-----------|---------------|
| TF-IDF max features | 10,000 |
| TF-IDF stop words | 'english' |
| Number of features | 7 |
| Classifier | `RandomForestClassifier` |
| Hparams | `n_estimators=100` |
| Objective | N/A |

---

## Baseline #3: Cross-Encoder (sentence-transformers)

**Type**: Sentence Transformers

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CrossEncoder (Limited Version)                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   INPUT:                                                            │
│   ├── input_ids: (B, seq_len) - "[CLS] query [SEP] passage [SEP]"   │
│   └── attention_mask: (B, seq_len)                                  │
│                                                                     │
│   ENCODER: sentence-transformers/all-MiniLM-L6-v2                   │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                                                             │   │
│   │  Layer 0  ████████████████  FROZEN (requires_grad=False)    │   │
│   │  Layer 1  ████████████████  FROZEN (requires_grad=False)    │   │
│   │  Layer 2  ████████████████  FROZEN (requires_grad=False)    │   │
│   │  Layer 3  ████████████████  FROZEN (requires_grad=False)    │   │
│   │  Layer 4  ░░░░░░░░░░░░░░░░  TRAINABLE                       │   │
│   │  Layer 5  ░░░░░░░░░░░░░░░░  TRAINABLE                       │   │
│   │                                                             │   │
│   │  Freezing logic:                                            │   │
│   │  for name, param in encoder.named_parameters():             │   │
│   │      if 'layer.4' not in name and 'layer.5' not in name     │   │
│   │         and 'pooler' not in name:                           │   │
│   │          param.requires_grad = False                        │   │
│   │                                                             │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│   [CLS] TOKEN EXTRACTION:                                           │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  outputs = encoder(input_ids, attention_mask)               │   │
│   │  cls_output = outputs.last_hidden_state[:, 0]  →  (B, 384)  │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│   DROPOUT (HEAVY):                                                  │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  dropout = nn.Dropout(0.5)                                  │   │
│   │  cls_output = dropout(cls_output)                           │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│   CLASSIFIER:                                                       │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  classifier = nn.Linear(384, 1)                             │   │
│   │  logits = classifier(cls_output)  →  (B, 1)                 │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│   OUTPUT: logits.squeeze(-1)  →  (B,)                               │
│                                                                     │
│   TRAINING:                                                         │
│   ├── Loss: nn.BCEWithLogitsLoss()                                  │
│   ├── Optimizer: AdamW(lr=2e-5, weight_decay=0.01)                  │
│   └── Max epochs: 5 (early stopping patience=3)                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Implementation Details

| Component | Specification |
|-----------|---------------|
| Base model | `sentence-transformers/all-MiniLM-L6-v2` |
| Hidden size | 384 |
| Num layers | 6 (layers 0-3 frozen, layers 4-5 trainable) |
| Dropout | **0.5** (heavy) |
| Output | `nn.Linear(384, 1)` |
| Loss | `BCEWithLogitsLoss` |
| Optimizer | `AdamW(lr=2e-5, weight_decay=0.01)` |
| Max epochs | **5** |
| Patience | 3 |

---

## Baseline #4: BERT Cross-Encoder (CLS Classification)

**Type**: HuggingFace BERT

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                BertCrossEncoder (Full Fine-Tuning)                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   INPUT:                                                            │
│   ├── input_ids: (B, seq_len) - "[CLS] query [SEP] passage [SEP]"   │
│   ├── attention_mask: (B, seq_len)                                  │
│   └── token_type_ids: (B, seq_len) - optional                       │
│                                                                     │
│   ENCODER: bert-base-uncased                                        │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                                                             │   │
│   │  Layer 0   ░░░░░░░░░░░░░░░░  TRAINABLE                      │   │
│   │  Layer 1   ░░░░░░░░░░░░░░░░  TRAINABLE                      │   │
│   │  Layer 2   ░░░░░░░░░░░░░░░░  TRAINABLE                      │   │
│   │  Layer 3   ░░░░░░░░░░░░░░░░  TRAINABLE                      │   │
│   │  Layer 4   ░░░░░░░░░░░░░░░░  TRAINABLE                      │   │
│   │  Layer 5   ░░░░░░░░░░░░░░░░  TRAINABLE  (ALL 12 LAYERS)     │   │
│   │  Layer 6   ░░░░░░░░░░░░░░░░  TRAINABLE                      │   │
│   │  Layer 7   ░░░░░░░░░░░░░░░░  TRAINABLE                      │   │
│   │  Layer 8   ░░░░░░░░░░░░░░░░  TRAINABLE                      │   │
│   │  Layer 9   ░░░░░░░░░░░░░░░░  TRAINABLE                      │   │
│   │  Layer 10  ░░░░░░░░░░░░░░░░  TRAINABLE                      │   │
│   │  Layer 11  ░░░░░░░░░░░░░░░░  TRAINABLE                      │   │
│   │                                                             │   │
│   │  Hidden size: 768                                           │   │
│   │  Total params: ~110M                                        │   │
│   │                                                             │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│   [CLS] TOKEN EXTRACTION:                                           │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  outputs = bert(input_ids, attention_mask, token_type_ids)  │   │
│   │  cls_output = outputs.last_hidden_state[:, 0, :] →  (B, 768)│   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│   CLASSIFIER (nn.Sequential):                                       │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  nn.Dropout(0.1),           # Light dropout                 │   │
│   │  nn.Linear(768, 2)          # 2-class output                │   │
│   │                                                             │   │
│   │  logits = classifier(cls_output)  →  (B, 2)                 │   │
│   │           ↓                                                 │   │
│   │  [logit_not_relevant, logit_relevant]                       │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│   OUTPUT: logits  →  (B, 2)                                         │
│                                                                     │
│   INFERENCE:                                                        │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  probs = F.softmax(logits, dim=1)                           │   │
│   │  relevance_score = probs[:, 1]  # P(relevant)               │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│   TRAINING:                                                         │
│   ├── Loss: nn.CrossEntropyLoss()                                   │
│   ├── Labels: 0 (not relevant) or 1 (relevant)                      │
│   ├── Optimizer: AdamW(lr=2e-5, weight_decay=0.01)                  │
│   ├── Gradient clipping: max_norm=1.0                               │
│   └── Max epochs: 30 (early stopping patience=2)                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Implementation Details

| Component | Specification |
|-----------|---------------|
| Base model | `bert-base-uncased` |
| Hidden size | 768 |
| Num layers | 12 (ALL trainable) |
| Dropout | **0.1** (light) |
| Output | `nn.Linear(768, 2)` - 2 classes |
| Loss | `CrossEntropyLoss` |
| Optimizer | `AdamW(lr=2e-5, weight_decay=0.01)` |
| Gradient clipping | `max_norm=1.0` |
| Max epochs | 30 |
| Patience | 2 |
| Max sequence length | 128 |

### Why 2-Class Classification?

```python
# Output: (B, 2) logits
logits = model(input_ids, attention_mask, token_type_ids)

# Class 0: not relevant
# Class 1: relevant

# Training
loss = CrossEntropyLoss()(logits, labels)  # labels ∈ {0, 1}

# Inference - get P(relevant)
probs = F.softmax(logits, dim=1)
relevance_score = probs[:, 1]
```

---

## Summary Comparison

| Aspect | Baseline #1 | Baseline #2 | Baseline #3 | Baseline #4 |
|--------|-------------|-------------|-------------|-------------|
| **Type** | Neural | Ensemble (RF) | Transformer | Transformer |
| **Embeddings** | GloVe 300D | TF-IDF + GloVe | MiniLM 384D | BERT 768D |
| **Trainable params** | ~100K | N/A | ~3M | ~110M |
| **Frozen layers** | Embeddings | N/A | Layers 0-3 | None |
| **Dropout** | 0.2 | N/A | 0.5 | 0.1 |
| **Output** | 1 score | 1 probability | 1 logit | 2 logits |
| **Loss** | MarginRanking | Gini/Entropy | BCE | CrossEntropy |
| **Max epochs** | 30 | N/A | 5 | 30 |

---

---


## Competition Rules

### Team Requirements
- **Team Size:** 2-4 students
- **Team Nickname:** Unique, registered before first submission
- **Submissions:** Maximum 3 per day

### Submission Format

Students submit a ZIP file containing `run_public.txt` and `run_private.txt` (and optional `predict.py`):
```
<query_id> Q0 <passage_id> <rank> <score> <run_name>
```

Example:
```
1048585 Q0 7187158 1 2.73 awesome_team
1048585 Q0 7187157 2 2.71 awesome_team
```

### Grading Scale (Based on PRIVATE Leaderboard)

| Performance | Points |
|-------------|--------|
| Above Baseline #4 (BERT CLS) | 30 pts |
| Above Baseline #3 (Cross-Encoder) | 26 pts |
| Above Baseline #2 (BM25+RF) | 22 pts |
| Above Baseline #1 (Neural Re-Ranker) | 18 pts |
| Below all baselines | < 18 pts |

**Bonus:** +2 pts for Top 3 finishers

### Allowed Resources
- ✅ Provided MS MARCO training data
- ✅ Pre-trained word embeddings (GloVe, etc.)
- ✅ Pre-trained language models (BERT, etc.)
- ✅ Google Colab (free tier)
- ❌ External training data
- ❌ Using dev/test labels during inference

### Anti-Cheating Measures

1. **Two-Phase Evaluation:** Public leaderboard ≠ final grade
2. **Code Submission Required:** Must reproduce results
3. **Submission Monitoring:** Patterns checked for collusion
4. **Code Similarity Check:** Plagiarism detection
5. **On Request:** Students must be ready to explain their code if asked

### Cheating = 0 Points
- Label probing (systematic submission to discover labels)
- Sharing code/predictions between teams
- Multiple CodaBench accounts
- Plagiarism

---

## Deliverables & Submission

**All submissions are made via CodaBench.** No email submissions.

### How Groups Submit

| Deliverable | Where to Submit | When |
|-------------|-----------------|------|
| `run_public.txt` & `run_private.txt` | CodaBench (upload ZIP) | During competition (max 3/day) |
| Code Package | CodaBench (upload ZIP) | By deadline |
| Brief Report | CodaBench (include in ZIP) | By deadline |

**Note:** Only ONE team member needs to submit to CodaBench. Use the same CodaBench account for all team submissions.

---

### Prediction Submission

Submit a CSV file (`submission.csv`) to Kaggle matching this format:

```
id,expected
1048585,7187158 7187157 7187156 ...
```
(Space-separated list of Passage IDs).

Generated automatically by `generate_submissions.py`.

### Brief Report (REPORT.pdf)

Include a 1-2 page PDF in your final ZIP describing:
- Your approach and model architecture
- Training procedure and hyperparameters
- Clever innovations
- Results and analysis

### LLM Usage Policy

LLMs (ChatGPT, Claude, etc.) ARE allowed, but:
- Must include `PROMPT.md` with the regeneration prompt
- Must understand and explain ALL submitted code
- On request, students must be able to explain their code

---

## Important Dates

| Event | Date |
|-------|------|
| Competition Opens | January 7, 2026 at 14:00 |
| Submissions Close | January 14, 2026 at 23:59 |
| Code + Report Due | January 14, 2026 at 23:59 |

---

## Quick Reference

| Item | Location |
|------|----------|
| Full Rules | `RULES.md` |
| Kaggle Setup | `kaggle_setup/SETUP_GUIDE.md` |
| Student Starter Kit | `starter_kit/` |
| Baseline Implementations | `baselines/` |
| Instructor Contact | fabrizio.silvestri@uniroma1.it |

