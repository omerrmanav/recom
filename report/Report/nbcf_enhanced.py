import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
import math
import time
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({'font.size': 10, 'font.family': 'sans-serif'})

print("=" * 65)
print("  ENHANCED NBCF IMPLEMENTATION WITH TEMPORAL DECAY & CONFIDENCE ANALYSIS")
print("=" * 65)

# =====================================================================
# 1. DATASET LOADING & PREPARATION
# =====================================================================
print("\n[1] Loading MovieLens-1M dataset...")

try:
    ratings_df = pd.read_csv(
        'ml-1m/ratings.dat', sep='::', engine='python', encoding='latin-1',
        names=['user_id', 'item_id', 'rating', 'timestamp']
    )
except FileNotFoundError:
    print("ERROR: 'ml-1m/ratings.dat' not found. Please verify the dataset path.")
    exit()

train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)
train_df = train_df.reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)

user_r = defaultdict(dict)
item_r = defaultdict(dict)
user_ts = defaultdict(dict)

for _, row in train_df.iterrows():
    user_r[row['user_id']][row['item_id']] = row['rating']
    item_r[row['item_id']][row['user_id']] = row['rating']
    user_ts[row['user_id']][row['item_id']] = row['timestamp']

R_values  = [1, 2, 3, 4, 5]
alpha     = 0.01
num_R     = len(R_values)

global_max_ts = train_df['timestamp'].max()
DECAY_LAMBDA  = 1e-7   # controls how fast older ratings decay in weight

print(f"  Users     : {len(user_r):,}")
print(f"  Items     : {len(item_r):,}")
print(f"  Ratings   : {len(train_df):,} (train)  |  {len(test_df):,} (test)")
print("  Ready.\n")

# =====================================================================
# 2. CORE ALGORITHM FUNCTIONS (Original + Enhancements)
# =====================================================================

def calc_item_prior(i, y, item_r, alpha=0.01, num_R=5):
    if i not in item_r: return 1.0 / num_R
    ratings = list(item_r[i].values())
    count_y = sum(1 for r in ratings if r == y)
    return (count_y + alpha) / (len(ratings) + num_R * alpha)

def calc_user_prior(u, y, user_r, alpha=0.01, num_R=5):
    if u not in user_r: return 1.0 / num_R
    ratings = list(user_r[u].values())
    count_y = sum(1 for r in ratings if r == y)
    return (count_y + alpha) / (len(ratings) + num_R * alpha)

def calc_ub_likelihood(i, y, j, k, item_r, alpha=0.01, num_R=5):
    users_iy = {u for u, r in item_r.get(i, {}).items() if r == y}
    match = sum(1 for u in users_iy if item_r.get(j, {}).get(u) == k)
    total = sum(1 for u in users_iy if u in item_r.get(j, {}))
    return (match + alpha) / (total + num_R * alpha)

def calc_ib_likelihood(u, y, v, k, user_r, alpha=0.01, num_R=5):
    items_uy = {it for it, r in user_r.get(u, {}).items() if r == y}
    match = sum(1 for it in items_uy if user_r.get(v, {}).get(it) == k)
    total = sum(1 for it in items_uy if it in user_r.get(v, {}))
    return (match + alpha) / (total + num_R * alpha)

def normalize_log_scores(log_scores_dict):
    max_log = max(log_scores_dict.values())
    probs = {y: math.exp(log_s - max_log) for y, log_s in log_scores_dict.items()}
    total = sum(probs.values())
    return {y: p / total for y, p in probs.items()}

def ub_score(u, i, user_r, item_r, alpha=0.01, num_R=5):
    log_scores = {}
    past_items = list(user_r.get(u, {}).items())
    for y in R_values:
        log_s = math.log(calc_item_prior(i, y, item_r, alpha, num_R))
        for j, k in past_items:
            lik = calc_ub_likelihood(i, y, j, k, item_r, alpha, num_R)
            log_s += math.log(lik) if lik > 0 else -1e9
        log_scores[y] = log_s
    return normalize_log_scores(log_scores)

def ib_score(u, i, user_r, item_r, alpha=0.01, num_R=5):
    log_scores = {}
    past_users = list(item_r.get(i, {}).items())
    for y in R_values:
        log_s = math.log(calc_user_prior(u, y, user_r, alpha, num_R))
        for v, k in past_users:
            lik = calc_ib_likelihood(u, y, v, k, user_r, alpha, num_R)
            log_s += math.log(lik) if lik > 0 else -1e9
        log_scores[y] = log_s
    return normalize_log_scores(log_scores)

def hybrid_score(u, i, user_r, item_r, alpha=0.01, num_R=5):
    ub_s = ub_score(u, i, user_r, item_r, alpha, num_R)
    ib_s = ib_score(u, i, user_r, item_r, alpha, num_R)
    n_Iu, n_Ui = len(user_r.get(u, {})), len(item_r.get(i, {}))
    w_ub, w_ib = 1.0 / (1 + n_Iu), 1.0 / (1 + n_Ui)
    hybrid = {}
    for y in R_values:
        ub_val, ib_val = max(ub_s[y], 1e-300), max(ib_s[y], 1e-300)
        hybrid[y] = (ub_val ** w_ub) * (ib_val ** w_ib)
    total = sum(hybrid.values())
    if total > 0:
        hybrid = {y: val / total for y, val in hybrid.items()}
    return hybrid, ub_s, ib_s

def predict_argmax(scores):
    return max(scores, key=scores.get)

def temporal_weight(user_id, item_id, global_max_ts, decay_lambda=DECAY_LAMBDA):
    ts = user_ts.get(user_id, {}).get(item_id, global_max_ts)
    age = global_max_ts - ts
    return math.exp(-decay_lambda * age)

def ub_score_temporal(u, i, user_r, item_r, alpha=0.01, num_R=5):
    """User-based score with temporal decay on past ratings."""
    log_scores = {}
    past_items = list(user_r.get(u, {}).items())
    for y in R_values:
        log_s = math.log(calc_item_prior(i, y, item_r, alpha, num_R))
        for j, k in past_items:
            w = temporal_weight(u, j, global_max_ts)
            lik = calc_ub_likelihood(i, y, j, k, item_r, alpha, num_R)
            log_s += w * (math.log(lik) if lik > 0 else -1e9)
        log_scores[y] = log_s
    return normalize_log_scores(log_scores)

def ib_score_temporal(u, i, user_r, item_r, alpha=0.01, num_R=5):
    """Item-based score with temporal decay on co-rater history."""
    log_scores = {}
    past_users = list(item_r.get(i, {}).items())
    for y in R_values:
        log_s = math.log(calc_user_prior(u, y, user_r, alpha, num_R))
        for v, k in past_users:
            ts_vi = user_ts.get(v, {}).get(i, global_max_ts)
            age   = global_max_ts - ts_vi
            w     = math.exp(-DECAY_LAMBDA * age)
            lik = calc_ib_likelihood(u, y, v, k, user_r, alpha, num_R)
            log_s += w * (math.log(lik) if lik > 0 else -1e9)
        log_scores[y] = log_s
    return normalize_log_scores(log_scores)

def hybrid_temporal(u, i, user_r, item_r, alpha=0.01, num_R=5):
    ub_s = ub_score_temporal(u, i, user_r, item_r, alpha, num_R)
    ib_s = ib_score_temporal(u, i, user_r, item_r, alpha, num_R)
    n_Iu, n_Ui = len(user_r.get(u, {})), len(item_r.get(i, {}))
    w_ub, w_ib = 1.0 / (1 + n_Iu), 1.0 / (1 + n_Ui)
    hybrid = {}
    for y in R_values:
        ub_val, ib_val = max(ub_s[y], 1e-300), max(ib_s[y], 1e-300)
        hybrid[y] = (ub_val ** w_ub) * (ib_val ** w_ib)
    total = sum(hybrid.values())
    if total > 0:
        hybrid = {y: val / total for y, val in hybrid.items()}
    return hybrid, ub_s, ib_s

def prediction_confidence(scores_dict):
    probs = list(scores_dict.values())
    entropy = -sum(p * math.log(p + 1e-12) for p in probs)
    max_entropy = math.log(len(probs))
    confidence = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0
    return confidence

# =====================================================================
# 3. EVALUATION (Original + Temporal + Confidence-filtered)
# =====================================================================
print("[2] Evaluating models (this may take a minute)...")

small_users = {u for u in user_r if 20 <= len(user_r[u]) <= 30}
eval_df = test_df[test_df['user_id'].isin(small_users)].head(1000)

errors_ub, errors_ib, errors_h, errors_ht = [], [], [], []
actuals, preds_h, preds_ht = [], [], []
confidences = []
latencies_h, latencies_ht = [], []

for _, row in eval_df.iterrows():
    u, i, true_r = int(row['user_id']), int(row['item_id']), int(row['rating'])

    t0 = time.perf_counter()
    h_s, ub_s, ib_s = hybrid_score(u, i, user_r, item_r)
    latencies_h.append(time.perf_counter() - t0)

    t1 = time.perf_counter()
    ht_s, ubt_s, ibt_s = hybrid_temporal(u, i, user_r, item_r)
    latencies_ht.append(time.perf_counter() - t1)

    p_ub  = predict_argmax(ub_s)
    p_ib  = predict_argmax(ib_s)
    p_h   = predict_argmax(h_s)
    p_ht  = predict_argmax(ht_s)

    conf  = prediction_confidence(ht_s)

    errors_ub.append(abs(true_r - p_ub))
    errors_ib.append(abs(true_r - p_ib))
    errors_h.append(abs(true_r - p_h))
    errors_ht.append(abs(true_r - p_ht))

    actuals.append(true_r)
    preds_h.append(p_h)
    preds_ht.append(p_ht)
    confidences.append(conf)

mae_ub      = np.mean(errors_ub)
mae_ib      = np.mean(errors_ib)
mae_hybrid  = np.mean(errors_h)
mae_temporal= np.mean(errors_ht)
print(f"  MAE — UB: {mae_ub:.4f}  IB: {mae_ib:.4f}  Hybrid: {mae_hybrid:.4f}  Temporal: {mae_temporal:.4f}")

prec_h  = precision_score(actuals, preds_h,  average='macro', zero_division=0)
rec_h   = recall_score(actuals, preds_h,  average='macro', zero_division=0)
f1_h    = f1_score(actuals, preds_h,  average='macro', zero_division=0)
prec_ht = precision_score(actuals, preds_ht, average='macro', zero_division=0)
rec_ht  = recall_score(actuals, preds_ht, average='macro', zero_division=0)
f1_ht   = f1_score(actuals, preds_ht, average='macro', zero_division=0)
print(f"  Hybrid   — P: {prec_h:.4f}  R: {rec_h:.4f}  F1: {f1_h:.4f}")
print(f"  Temporal — P: {prec_ht:.4f}  R: {rec_ht:.4f}  F1: {f1_ht:.4f}")

act_bin  = [0 if a <= 3 else 1 for a in actuals]
pred_bin = [0 if p <= 3 else 1 for p in preds_ht]
cm_bin   = confusion_matrix(act_bin, pred_bin, labels=[0, 1])

conf_threshold = 0.3
high_conf_mask = [c >= conf_threshold for c in confidences]
coverage = sum(high_conf_mask) / len(high_conf_mask)
mae_filtered = np.mean([e for e, m in zip(errors_ht, high_conf_mask) if m]) if any(high_conf_mask) else 0
print(f"  High-confidence coverage ({conf_threshold}): {coverage:.1%}  Filtered MAE: {mae_filtered:.4f}\n")

# =====================================================================
# 4. SUSTAINABILITY: Computational Profiling
# =====================================================================
avg_lat_h  = np.mean(latencies_h) * 1000
avg_lat_ht = np.mean(latencies_ht) * 1000
p95_lat_h  = np.percentile(latencies_h, 95) * 1000
p95_lat_ht = np.percentile(latencies_ht, 95) * 1000

user_sizes = [len(user_r[u]) for u in list(user_r.keys())[:200]]
item_sizes = [len(item_r[i]) for i in list(item_r.keys())[:200]]

print("[3] Sustainability / Efficiency stats:")
print(f"  Hybrid latency   — avg: {avg_lat_h:.1f} ms  p95: {p95_lat_h:.1f} ms")
print(f"  Temporal latency — avg: {avg_lat_ht:.1f} ms  p95: {p95_lat_ht:.1f} ms")

# =====================================================================
# 5. VISUALIZATION — 8 PLOTS
# =====================================================================
print("\n[4] Generating all visualisations...")

BLUE      = "#1a4f8a"
LIGHT_BLU = "#4a90d9"
ORANGE    = "#e07b39"
GREEN     = "#2e8b5a"
GRAY      = "#888888"
BG        = "#f8f9fc"

def styled_ax(ax, title, xlabel="", ylabel=""):
    ax.set_facecolor(BG)
    ax.set_title(title, fontweight='bold', color=BLUE, pad=8, fontsize=10)
    if xlabel: ax.set_xlabel(xlabel, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines[['top','right']].set_visible(False)

def draw_academic_table(ax, df, title):
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('black')
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#d9e8f5')
            cell.visible_edges = 'BT'
            cell.set_linewidth(2)
        else:
            cell.visible_edges = ''
            if row == len(df):
                cell.visible_edges = 'B'
                cell.set_linewidth(2)
    ax.set_title(title, fontweight='bold', color=BLUE, loc='left', pad=10, fontsize=10)

# ══════════════════════════════════════════════════════════════════════
# FIGURE A: Original 4 Tables (Table 1–5) with improvements
# ══════════════════════════════════════════════════════════════════════
target_u = test_df.iloc[0]['user_id']
target_i = test_df.iloc[0]['item_id']
u_items  = list(user_r[target_u].keys())[:4]
other_items = [i for i in item_r.keys() if i not in user_r[target_u] and i != target_i][:4]
selected_items = [target_i] + u_items + other_items
i_users = [u for u in item_r[target_i].keys() if u != target_u][:4]
selected_users = [target_u] + i_users

fig, ax = plt.subplots(figsize=(10, 3))
mat_data = []
for u in selected_users:
    row = [f"u_{u}"]
    for i in selected_items:
        val = user_r[u].get(i, '•')
        row.append(val)
    mat_data.append(row)
df1 = pd.DataFrame(mat_data, columns=[""] + [f"i_{i}" for i in selected_items])
draw_academic_table(ax, df1, "TABLE 1. Running example of the rating matrix (ML-1M).")
plt.tight_layout()
plt.savefig('outputs/table1_rating_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
ub_priors = [calc_item_prior(target_i, y, item_r) for y in R_values]
ib_priors = [calc_user_prior(target_u, y, user_r) for y in R_values]
df2 = pd.DataFrame([
    ["user-based"] + [f"{v:.6f}" for v in ub_priors],
    ["item-based"] + [f"{v:.6f}" for v in ib_priors]
], columns=[""] + R_values)
draw_academic_table(ax1, df2, f"TABLE 2. Prior probabilities (Item {target_i}, User {target_u}).")

t3_data = []
for j in u_items:
    row = [f"i_{j}"]
    for y in R_values:
        val = calc_ub_likelihood(target_i, y, j, user_r[target_u][j], item_r)
        row.append(f"{val:.6f}")
    t3_data.append(row)
df3 = pd.DataFrame(t3_data, columns=["y"] + R_values)
draw_academic_table(ax2, df3, f"TABLE 3. User-based likelihood for item {target_i}.")
plt.tight_layout()
plt.savefig('outputs/table2_3_priors_likelihood.png', dpi=150, bbox_inches='tight')
plt.show()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
t4_data = []
for v in i_users:
    row = [f"u_{v}"]
    for y in R_values:
        val = calc_ib_likelihood(target_u, y, v, item_r[target_i][v], user_r)
        row.append(f"{val:.8f}")
    t4_data.append(row)
df4 = pd.DataFrame(t4_data, columns=["v"] + R_values)
draw_academic_table(ax1, df4, f"TABLE 4. Item-based likelihood for user {target_u}.")

h_s, ub_s, ib_s   = hybrid_score(target_u, target_i, user_r, item_r)
ht_s, ubt_s, ibt_s = hybrid_temporal(target_u, target_i, user_r, item_r)

def format_row(name, scores_dict):
    max_k = predict_argmax(scores_dict)
    row = [name]
    for y in R_values:
        val_str = f"{scores_dict[y]:.5f}"
        if y == max_k: val_str = f"*{val_str}*"
        row.append(val_str)
    return row

df5 = pd.DataFrame([
    format_row("user-based",  ub_s),
    format_row("item-based",  ib_s),
    format_row("Hybrid",      h_s),
    format_row("Temporal-UB", ubt_s),
    format_row("Temporal-IB", ibt_s),
    format_row("Hybrid-Temp", ht_s),
], columns=["Method"] + R_values)
draw_academic_table(ax2, df5, "TABLE 5. Classification scores — original vs. temporal variants (* = predicted class).")
plt.tight_layout()
plt.savefig('outputs/table4_5_ib_scores.png', dpi=150, bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════════════════
# FIGURE B: Confusion Matrices (METHOD)
# ══════════════════════════════════════════════════════════════════════
cm        = confusion_matrix(actuals, preds_ht, labels=R_values)
cm_norm   = cm.astype('float') / cm.sum(axis=1, keepdims=True)
cm_norm   = np.nan_to_num(cm_norm)
cm_bin_norm = cm_bin.astype('float') / cm_bin.sum(axis=1, keepdims=True)
cm_bin_norm = np.nan_to_num(cm_bin_norm)

fig, axes = plt.subplots(2, 2, figsize=(11, 9))
fig.suptitle("FIGURE 1. Confusion Matrices — Temporal Hybrid NBCF (ML-1M)",
             fontweight='bold', color=BLUE, fontsize=12, y=0.98)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=R_values, yticklabels=R_values, ax=axes[0,0])
axes[0,0].set_title("a) 5-Class — Raw Counts", fontsize=10)
axes[0,0].set_ylabel("Actual"); axes[0,0].set_xlabel("Predicted")

sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Blues', cbar=True,
            xticklabels=R_values, yticklabels=R_values, ax=axes[0,1])
axes[0,1].set_title("b) 5-Class — Normalised", fontsize=10)
axes[0,1].set_ylabel("Actual"); axes[0,1].set_xlabel("Predicted")

sns.heatmap(cm_bin, annot=True, fmt='d', cmap='Oranges', cbar=True,
            xticklabels=['Not Like', 'Like'], yticklabels=['Not Like', 'Like'], ax=axes[1,0])
axes[1,0].set_title("c) Binary — Raw Counts", fontsize=10)
axes[1,0].set_ylabel("Actual"); axes[1,0].set_xlabel("Predicted")

sns.heatmap(cm_bin_norm, annot=True, fmt='.1%', cmap='Oranges', cbar=True,
            xticklabels=['Not Like', 'Like'], yticklabels=['Not Like', 'Like'], ax=axes[1,1])
axes[1,1].set_title("d) Binary — Normalised", fontsize=10)
axes[1,1].set_ylabel("Actual"); axes[1,1].set_xlabel("Predicted")

plt.tight_layout()
plt.savefig('outputs/figure1_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════════════════
# FIGURE C: MAE Comparison — 4 methods (METHOD + NOVELTY)
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 4))
methods = ['UB', 'IB', 'Hybrid', 'Hybrid\n+Temporal']
maes    = [mae_ub, mae_ib, mae_hybrid, mae_temporal]
colors  = [GRAY, GRAY, LIGHT_BLU, ORANGE]
bars = ax.bar(methods, maes, color=colors, width=0.5, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, maes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{val:.4f}", ha='center', fontsize=9, fontweight='bold')
ax.set_ylim(0, max(maes) * 1.25)
styled_ax(ax, "FIGURE 2. MAE Comparison — Original vs. Temporal NBCF", "Method", "MAE")
patches = [mpatches.Patch(color=LIGHT_BLU, label='Original Hybrid'),
           mpatches.Patch(color=ORANGE, label='Temporal Hybrid (NEW)')]
ax.legend(handles=patches, fontsize=9)
plt.tight_layout()
plt.savefig('outputs/figure2_mae_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════════════════
# FIGURE D: Per-Class F1 Score (METHOD)
# ══════════════════════════════════════════════════════════════════════
f1_per_class_h  = f1_score(actuals, preds_h,  average=None, labels=R_values, zero_division=0)
f1_per_class_ht = f1_score(actuals, preds_ht, average=None, labels=R_values, zero_division=0)

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(R_values))
w = 0.35
ax.bar(x - w/2, f1_per_class_h,  w, label='Hybrid (original)', color=LIGHT_BLU, edgecolor='white')
ax.bar(x + w/2, f1_per_class_ht, w, label='Hybrid + Temporal',  color=ORANGE,    edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels([f"Rating {r}" for r in R_values])
styled_ax(ax, "FIGURE 3. Per-Class F1 Score by Rating Value", "Rating Class", "F1 Score")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('outputs/figure3_per_class_f1.png', dpi=150, bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════════════════
# FIGURE E: Confidence Distribution + MAE vs Confidence (NOVELTY)
# ══════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

ax1.hist(confidences, bins=20, color=LIGHT_BLU, edgecolor='white', alpha=0.85)
ax1.axvline(conf_threshold, color=ORANGE, linestyle='--', lw=2, label=f'Threshold={conf_threshold}')
styled_ax(ax1, "FIGURE 4a. Prediction Confidence Distribution", "Confidence Score", "Count")
ax1.legend(fontsize=9)

buckets = np.linspace(0, 1, 11)
bucket_mae = []
bucket_cov = []
for lo, hi in zip(buckets[:-1], buckets[1:]):
    mask = [lo <= c < hi for c in confidences]
    errs = [e for e, m in zip(errors_ht, mask) if m]
    bucket_mae.append(np.mean(errs) if errs else np.nan)
    bucket_cov.append(sum(mask))

mids = (buckets[:-1] + buckets[1:]) / 2
valid = ~np.isnan(bucket_mae)
ax2.plot(mids[valid], np.array(bucket_mae)[valid], 'o-', color=ORANGE, lw=2, ms=6)
ax2.set_xlim(0, 1)
styled_ax(ax2, "FIGURE 4b. MAE vs. Prediction Confidence", "Confidence Bucket", "Mean MAE")
plt.tight_layout()
plt.savefig('outputs/figure4_confidence.png', dpi=150, bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════════════════
# FIGURE F: Rating Distribution — actual vs predicted (IMPACT)
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

act_counts  = [actuals.count(r) for r in R_values]
pred_h_cnt  = [preds_h.count(r)  for r in R_values]
pred_ht_cnt = [preds_ht.count(r) for r in R_values]

x = np.arange(len(R_values))
w = 0.28
axes[0].bar(x - w, act_counts,  w, label='Actual',  color=GRAY,      edgecolor='white')
axes[0].bar(x,     pred_h_cnt,  w, label='Hybrid',  color=LIGHT_BLU, edgecolor='white')
axes[0].bar(x + w, pred_ht_cnt, w, label='Temporal',color=ORANGE,    edgecolor='white')
axes[0].set_xticks(x); axes[0].set_xticklabels(R_values)
styled_ax(axes[0], "FIGURE 5a. Predicted vs Actual Rating Distribution", "Rating", "Count")
axes[0].legend(fontsize=9)

bias_h  = [np.mean([e for a, p, e in zip(actuals, preds_h,  errors_h)  if a == r]) for r in R_values]
bias_ht = [np.mean([e for a, p, e in zip(actuals, preds_ht, errors_ht) if a == r]) for r in R_values]
axes[1].plot(R_values, bias_h,  'o-', color=LIGHT_BLU, lw=2, ms=7, label='Hybrid')
axes[1].plot(R_values, bias_ht, 's-', color=ORANGE,    lw=2, ms=7, label='Temporal')
styled_ax(axes[1], "FIGURE 5b. Mean Absolute Error per Rating Class", "Actual Rating", "MAE")
axes[1].legend(fontsize=9)
plt.tight_layout()
plt.savefig('outputs/figure5_rating_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════════════════
# FIGURE G: Sustainability — Latency Distribution (SUSTAINABILITY)
# ══════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

ax1.hist([l*1000 for l in latencies_h],  bins=30, alpha=0.7, color=LIGHT_BLU, label='Hybrid', edgecolor='white')
ax1.hist([l*1000 for l in latencies_ht], bins=30, alpha=0.7, color=ORANGE,    label='Temporal', edgecolor='white')
ax1.axvline(avg_lat_h,  color=BLUE,   linestyle='--', lw=1.5, label=f'avg H={avg_lat_h:.1f}ms')
ax1.axvline(avg_lat_ht, color='#b84c00', linestyle='--', lw=1.5, label=f'avg T={avg_lat_ht:.1f}ms')
styled_ax(ax1, "FIGURE 6a. Prediction Latency Distribution", "Latency (ms)", "Count")
ax1.legend(fontsize=8)

depth_bins = [0, 5, 10, 20, 30, 50]
tput_h, tput_ht = [], []
for lo, hi in zip(depth_bins[:-1], depth_bins[1:]):
    mask = [lo <= len(user_r.get(u, {})) < hi for u in eval_df['user_id'].astype(int)]
    lats_h  = [l for l, m in zip(latencies_h,  mask) if m]
    lats_ht = [l for l, m in zip(latencies_ht, mask) if m]
    tput_h.append(1/np.mean(lats_h)  if lats_h  else 0)
    tput_ht.append(1/np.mean(lats_ht) if lats_ht else 0)

mids = [(lo+hi)/2 for lo, hi in zip(depth_bins[:-1], depth_bins[1:])]
ax2.plot(mids, tput_h,  'o-', color=LIGHT_BLU, lw=2, ms=7, label='Hybrid')
ax2.plot(mids, tput_ht, 's-', color=ORANGE,    lw=2, ms=7, label='Temporal')
styled_ax(ax2, "FIGURE 6b. Throughput vs. User Rating History Depth", "Avg History Size", "Predictions/sec")
ax2.legend(fontsize=9)
plt.tight_layout()
plt.savefig('outputs/figure6_sustainability.png', dpi=150, bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════════════════
# FIGURE H: Full Metrics Summary Table (METHOD + IMPACT)
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 4))
df_summary = pd.DataFrame({
    "Method": ["NBCF (user-based)", "NBCF (item-based)",
                "NBCF (hybrid)", "NBCF (hybrid + temporal decay) [NEW]"],
    "MAE":        [f"{mae_ub:.5f}", f"{mae_ib:.5f}", f"{mae_hybrid:.5f}", f"{mae_temporal:.5f}"],
    "Precision":  ["—", "—", f"{prec_h:.5f}", f"{prec_ht:.5f}"],
    "Recall":     ["—", "—", f"{rec_h:.5f}",  f"{rec_ht:.5f}"],
    "F1 (macro)": ["—", "—", f"{f1_h:.5f}",   f"{f1_ht:.5f}"],
    "Coverage":   ["—", "—", "—",              f"{coverage:.1%}"],
})
draw_academic_table(ax, df_summary, "TABLE 7. Comprehensive evaluation metrics for all NBCF variants.")
plt.tight_layout()
plt.savefig('outputs/table7_full_metrics.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*65)
print("  All figures saved to outputs/")
print("  Script completed successfully.")
print("="*65)