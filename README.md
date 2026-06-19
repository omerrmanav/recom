# NBCF Recommender System — Final Project

**Course:** Recommender Systems  
**Group Members:**
- Ömer MANAV — 210209003
- Kaan Bahadır SAĞRA — 210201011

---

## Project Overview

This project implements a **Naïve Bayes Collaborative Filtering (NBCF)** recommender system on the MovieLens-1M dataset. The final version extends the midterm implementation with Top-K neighborhood selection, fixed confidence scoring, RMSE evaluation, user-group stratification, and a geometric hybrid model.

---

## Dataset

The MovieLens-1M dataset is already included in the repository under `report/final/ml-1m/`. No separate download is needed.

Files used:
- `ml-1m/ratings.dat` — 1,000,209 ratings (UserID::MovieID::Rating::Timestamp)
- `ml-1m/movies.dat` — 3,706 movies
- `ml-1m/users.dat` — 6,040 users

---

## Installation

Python 3.8+ is required.

### 1. Create a virtual environment

```bash
cd report/final
python3 -m venv venv
```

### 2. Activate the virtual environment

```bash
# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Code

Navigate to `report/final/` and run the main script:

```bash
cd report/final
MPLBACKEND=Agg python nbcf_final.py
```

> **Note:** `MPLBACKEND=Agg` is required on headless (no display) environments. On a desktop system with a display, you can omit it and plots will render in a window.

The script will:
1. Load and split the MovieLens-1M dataset (80/20 train/test)
2. Train User-Based and Item-Based NBCF models
3. Evaluate on 1,000 test samples (main), 300 per user group, and 200 per alpha value
4. Save all outputs to `report/final/outputs/`

Expected runtime: approximately 3–8 minutes depending on hardware.

---

## Output Files

All figures and tables are saved to `report/final/outputs/`:

| File | Description |
|------|-------------|
| `figure1_confusion_matrices.png` | Confusion matrices (UB / IB, raw / normalized) |
| `figure2_mae_rmse_comparison.png` | MAE and RMSE bar chart across model variants |
| `figure3_per_class_f1.png` | Per-class F1 scores |
| `figure4_confidence.png` | Confidence score distribution |
| `figure5_rating_distribution.png` | Rating distribution in train/test sets |
| `figure6_sustainability.png` | Compute time vs. accuracy trade-off |
| `figure7_user_groups.png` | MAE by user activity group (Light / Medium / Heavy) |
| `figure8_confidence_sweep.png` | Accuracy vs. confidence threshold sweep |
| `figure9_alpha_sensitivity.png` | MAE sensitivity to Laplace smoothing alpha |
| `table1_rating_matrix.png` | Sample rating matrix |
| `table2_3_priors_likelihood.png` | Prior and likelihood tables |
| `table4_5_ib_scores.png` | Item-based score tables |
| `table6_user_groups.png` | Metric table by user group |
| `table7_full_metrics.png` | Full metric comparison table |

---

## Generating Reports

To regenerate the Word documents (Report_1, Report_2, Report_3) and INFO_REPORT.txt:

```bash
cd report/final
python generate_reports.py
```

The following files will be created in `report/final/`:
- `Report_1_Final.docx` — Full academic paper with embedded figures
- `Report_2_Final.docx` — Midterm vs. final differences summary
- `Report_3_Final.docx` — Line-by-line code and output explanation
- `INFO_REPORT.txt` — Turkish reference document

---

## Project Structure

```
recommender/
├── README.md
├── final_project_guide.txt
└── report/
    └── final/
        ├── nbcf_final.py          # Main implementation
        ├── generate_reports.py    # Report generation script
        ├── requirements.txt       # Python dependencies
        ├── ml-1m/                 # MovieLens-1M dataset
        │   ├── ratings.dat
        │   ├── movies.dat
        │   └── users.dat
        ├── outputs/               # Generated figures and tables
        ├── Report_1_Final.docx
        ├── Report_2_Final.docx
        └── Report_3_Final.docx
```
