# NBCF Recommender System — Final Project

**Course:** Recommender Systems 
**Group Members:**
- Ömer MANAV — 210209003
- Kaan Bahadır SAĞRA — 210201011

---

## About

For our final project we extended the midterm NBCF implementation. We added Top-K neighbor selection, fixed the confidence scoring (it was giving 0% coverage before), added RMSE as a second metric, and tested the model across different user activity groups.

Dataset is MovieLens-1M. Already included in the repo, no need to download anything.

---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Running

```bash
python nbcf_final.py
```

Takes around 3–8 minutes. Outputs are saved to `outputs/`.

---

## Outputs

9 figures and 7 tables saved to `outputs/`:

- figure1 — confusion matrices
- figure2 — MAE/RMSE comparison
- figure3 — per-class F1
- figure4 — confidence score distribution
- figure5 — rating distribution
- figure6 — latency comparison
- figure7 — results by user group
- figure8 — confidence threshold sweep
- figure9 — alpha sensitivity
- table1 — rating matrix sample
- table2/3 — prior and likelihood tables
- table4/5 — item-based score tables
- table6 — user group metrics
- table7 — full metrics summary

---

## File Structure

```
submission/
├── README.md
├── nbcf_final.py
├── requirements.txt
├── ml-1m/
│   ├── ratings.dat
│   ├── movies.dat
│   └── users.dat
├── outputs/               (created when you run the script)
├── Report_1_Final.pdf
├── Report_2_Final.pdf
└── Report_3_Final.pdf
```
