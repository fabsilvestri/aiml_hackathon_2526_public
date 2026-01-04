# 🎓 AIML Hackathon 2526 - Student Guide: Zero to Hero

This guide provides the complete step-by-step procedure to participate in the competition, set up your environment, and submit your first run.

## 📁 Submission Format (Critical)

You must submit a CSV file with exactly **4912 rows** (plus header).

**Header:** `id,expected`
**Columns:**
- `id`: The Query ID (string). Matches `test.csv`.
- `expected`: A space-separated list of Passage IDs (pids) ranked by relevance.

**Example content:**
```csv
id,expected
1001,70231 23423 52342 8842 1102
1002,9921 5521 112 334 55
...
```

---

## 🚀 Step-by-Step Procedure

### 1. Join the Competition
- Use the Invite Link provided in Google Classroom.
- **Create Team**: Go to the **Team** tab. Set your **Team Name** to match your Google Form registration EXACTLY. invite your teammates.

### 2. Setup Kaggle Notebook
1.  **New Notebook**: Click "Code" -> "New Notebook".
2.  Copy the code cells below into your notebook.

### 3. Generate Your First Submission (Random Baseline)

Copy and run the following cells in your Kaggle notebook:

**Cell 1: Install Dependencies**
```python
!pip install torch transformers sentence-transformers scikit-learn pandas numpy tqdm rank_bm25 pyarrow -q
```

**Cell 2: Download Dataset**
```python
import os
import zipfile
import urllib.request

DATASET_URL = "https://github.com/fabsilvestri/aiml_hackathon_data/releases/download/v1.0/kaggle_data.zip"
DATA_DIR = "data"

print("Downloading dataset...")
urllib.request.urlretrieve(DATASET_URL, "kaggle_data.zip")
os.makedirs(DATA_DIR, exist_ok=True)
with zipfile.ZipFile("kaggle_data.zip", "r") as zf:
    zf.extractall(DATA_DIR)
os.remove("kaggle_data.zip")
print("Dataset ready!")
```

**Cell 3: Load Data**
```python
import pandas as pd
import random

test_queries = pd.read_csv(f"{DATA_DIR}/test.csv")
collection = pd.read_parquet(f"{DATA_DIR}/collection.parquet")
all_pids = collection["pid"].astype(str).tolist()

print(f"Loaded {len(test_queries)} test queries and {len(all_pids)} passages.")
```

**Cell 4: Generate Random Baseline**
```python
results = []
for qid in test_queries["id"]:
    ranked_pids = random.sample(all_pids, 10)
    results.append({"id": str(qid), "expected": " ".join(ranked_pids)})

submission = pd.DataFrame(results)
submission.to_csv("submission.csv", index=False)
print(f"Created submission.csv with {len(submission)} rows.")
```

**Output:** `submission.csv` is ready for upload!

### 4. Submit to Leaderboard
1.  In the notebook output or file browser, find `submission.csv`.
2.  Download the file.
3.  Go to the Competition Page -> **Submit Predictions**.
4.  Upload the file.
5.  Check your score on the **Public Leaderboard**.

### 5. Improve & Iterate
- Implement your own retrieval models (e.g. BM25, Neural, or Transformers).
- Use the `val.parquet` data to evaluate locally before submitting.
- Generate new submissions and upload to climb the rankings!

---

## ❓ FAQ

**Q: My submission fails with "Submission must have 4912 rows".**
A: You likely generated predictions for all queries without filtering. Use the provided `generate_submissions.py` script which handles filtering automatically against `test.csv`.

**Q: Local vs Leaderboard Score?**
A: Both use **MAP@10**. However, the Leaderboard uses the **Private Test Set** (hidden from you). Your local validation uses a smaller `val.parquet` set. Slight differences are expected, but they should correlate.

**Q: "Unexpected Column" error?**
A: Ensure your CSV header is lowercase `id,expected`. The provided scripts do this automatically.
