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
2.  **Clone Repository**: In the first cell, run:
    ```python
    !git clone https://github.com/fabsilvestri/aiml_hackathon_2526_public.git
    %cd aiml_hackathon_2526_public
    ```

### 3. Generate Your First Submission (Random Baseline)
Instead of writing code from scratch, we have provided a ready-to-use notebook.

1.  Open `starter_notebook.ipynb` from the file list.
2.  **Run All Cells**: This notebook will automatically:
    *   Install all necessary dependencies.
    *   Load the dataset (detecting if you are on Kaggle or local).
    *   Generate a **Random Baseline** and save it to `submission_files/submission_random.csv`.

*Note: You can verify the output in the `submission_files` directory.*

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
