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
2.  **Add Data**: Click "Add Input" -> Search for the competition dataset (e.g., `aiml-hackathon-2526-data` or similar provided name). Add it.
3.  **Clone Repository**: In the first cell, run:
    ```python
    !git clone https://github.com/fabsilvestri/aiml_hackathon_2526_public.git
    %cd aiml_hackathon_2526_public
    !pip install -r requirements.txt
    ```

### 3. Generate Your First Submission (Random Baseline)
Since baseline implementations are secret, you should start by creating a simple random submission to verify your pipeline. Copy the following code into a cell in your notebook:

```python
import pandas as pd
import random
import os

# 1. Define Paths (Kaggle or Local)
# On Kaggle, dataset is usually at /kaggle/input/your-dataset-name/
# Just in case, check where files are:
if os.path.exists("/kaggle/input"):
    # START HERE: Adapting paths!
    # You might need to adjust "aiml-hackathon-2526" to your actual dataset directory name
    DATA_DIR = "/kaggle/input/aiml-hackathon-2526/msmarco_sampled" 
    TEST_FILE = "/kaggle/input/aiml-hackathon-2526/test.csv"
else:
    # Local fallback
    DATA_DIR = "msmarco_sampled"
    TEST_FILE = "test.csv"

print(f"Using Data Dir: {DATA_DIR}")

# 2. Load Queries and Collection
test_queries = pd.read_csv(TEST_FILE)
collection = pd.read_parquet(f"{DATA_DIR}/collection.parquet")
all_pids = collection['pid'].astype(str).tolist()

print(f"Loaded {len(test_queries)} queries and {len(all_pids)} passages.")

# 3. Generate Random Rankings
results = []
for qid in test_queries['id']:
    # Randomly sample 10 PIDs from the collection
    ranked_pids = random.sample(all_pids, 10)
    results.append({
        'id': str(qid),
        'expected': " ".join(ranked_pids)
    })

# 4. Save Submission
os.makedirs("submission_files", exist_ok=True)
submission = pd.DataFrame(results)
submission.to_csv("submission_files/submission_random.csv", index=False)
print("Created submission_files/submission_random.csv with", len(submission), "rows.")
```

Run this cell. It will create a valid (but poor performing) generic submission file.

### 4. Submit to Leaderboard
1.  In the Output file browser (right sidebar), find `submission_files`.
2.  Download `submission_random.csv` (or your own file).
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
