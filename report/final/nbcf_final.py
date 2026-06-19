import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import math
import time
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

os.makedirs('outputs', exist_ok=True)

plt.rcParams.update({'font.size': 10, 'font.family': 'sans-serif'})

BLUE      = "#1a4f8a"
LIGHT_BLU = "#4a90d9"
ORANGE    = "#e07b39"
GREEN     = "#2e8b5a"
PURPLE    = "#7b2d8b"
GRAY      = "#888888"
BG        = "#f8f9fc"

print("=" * 72)
print("  FINAL NBCF: TOP-K NEIGHBORHOOD | RMSE | USER GROUPS | ALPHA SWEEP")
print("=" * 72)

# =====================================================================
# GLOBAL CONSTANTS
# =====================================================================
R_values     = [1, 2, 3, 4, 5]
ALPHA        = 0.01
NUM_R        = len(R_values)
DECAY_LAMBDA = 1e-7
TOP_K        = 30

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
    print("ERROR: 'ml-1m/ratings.dat' not found.")
    exit()

train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)
train_df = train_df.reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)

user_r  = defaultdict(dict)
item_r  = defaultdict(dict)
user_ts = defaultdict(dict)

for _, row in train_df.iterrows():
    user_r[row['user_id']][row['item_id']] = row['rating']
    item_r[row['item_id']][row['user_id']] = row['rating']
    user_ts[row['user_id']][row['item_id']] = row['timestamp']

global_max_ts = train_df['timestamp'].max()

# Precompute frozen sets for O(1) intersection in Top-K selection
item_users_set = {i: frozenset(u_dict.keys()) for i, u_dict in item_r.items()}
user_items_set = {u: frozenset(i_dict.keys()) for u, i_dict in user_r.items()}

print(f"  Users     : {len(user_r):,}")
print(f"  Items     : {len(item_r):,}")
print(f"  Ratings   : {len(train_df):,} train | {len(test_df):,} test")
print(f"  K         : {TOP_K}")
print("  Ready.\n")

# =====================================================================
# 2. CORE ALGORITHM FUNCTIONS  (unchanged from midterm)
# =====================================================================

def calc_item_prior(i, y, a=ALPHA):
    if i not in item_r:
        return 1.0 / NUM_R
    ratings = list(item_r[i].values())
    return (sum(1 for r in ratings if r == y) + a) / (len(ratings) + NUM_R * a)

def calc_user_prior(u, y, a=ALPHA):
    if u not in user_r:
        return 1.0 / NUM_R
    ratings = list(user_r[u].values())
    return (sum(1 for r in ratings if r == y) + a) / (len(ratings) + NUM_R * a)

def calc_ub_likelihood(i, y, j, k, a=ALPHA):
    users_iy = {u for u, r in item_r.get(i, {}).items() if r == y}
    match    = sum(1 for u in users_iy if item_r.get(j, {}).get(u) == k)
    total    = sum(1 for u in users_iy if u in item_r.get(j, {}))
    return (match + a) / (total + NUM_R * a)

def calc_ib_likelihood(u, y, v, k, a=ALPHA):
    items_uy = {it for it, r in user_r.get(u, {}).items() if r == y}
    match    = sum(1 for it in items_uy if user_r.get(v, {}).get(it) == k)
    total    = sum(1 for it in items_uy if it in user_r.get(v, {}))
    return (match + a) / (total + NUM_R * a)

def normalize_log_scores(log_scores_dict):
    max_log = max(log_scores_dict.values())
    probs   = {y: math.exp(ls - max_log) for y, ls in log_scores_dict.items()}
    total   = sum(probs.values())
    return {y: p / total for y, p in probs.items()}

def ub_score(u, i, a=ALPHA):
    log_scores = {}
    past_items = list(user_r.get(u, {}).items())
    for y in R_values:
        log_s = math.log(calc_item_prior(i, y, a))
        for j, k in past_items:
            lik    = calc_ub_likelihood(i, y, j, k, a)
            log_s += math.log(lik) if lik > 0 else -1e9
        log_scores[y] = log_s
    return normalize_log_scores(log_scores)

def ib_score(u, i, a=ALPHA):
    log_scores = {}
    past_users = list(item_r.get(i, {}).items())
    for y in R_values:
        log_s = math.log(calc_user_prior(u, y, a))
        for v, k in past_users:
            lik    = calc_ib_likelihood(u, y, v, k, a)
            log_s += math.log(lik) if lik > 0 else -1e9
        log_scores[y] = log_s
    return normalize_log_scores(log_scores)

def hybrid_score(u, i):
    ub_s = ub_score(u, i)
    ib_s = ib_score(u, i)
    n_Iu, n_Ui = len(user_r.get(u, {})), len(item_r.get(i, {}))
    w_ub, w_ib = 1.0 / (1 + n_Iu), 1.0 / (1 + n_Ui)
    hybrid = {y: max(ub_s[y], 1e-300) ** w_ub * max(ib_s[y], 1e-300) ** w_ib
              for y in R_values}
    total = sum(hybrid.values())
    return {y: v / total for y, v in hybrid.items()}, ub_s, ib_s

def predict_argmax(scores):
    return max(scores, key=scores.get)

def temporal_weight(user_id, item_id):
    ts = user_ts.get(user_id, {}).get(item_id, global_max_ts)
    return math.exp(-DECAY_LAMBDA * (global_max_ts - ts))

def ub_score_temporal(u, i, a=ALPHA):
    log_scores = {}
    past_items = list(user_r.get(u, {}).items())
    for y in R_values:
        log_s = math.log(calc_item_prior(i, y, a))
        for j, k in past_items:
            w      = temporal_weight(u, j)
            lik    = calc_ub_likelihood(i, y, j, k, a)
            log_s += w * (math.log(lik) if lik > 0 else -1e9)
        log_scores[y] = log_s
    return normalize_log_scores(log_scores)

def ib_score_temporal(u, i, a=ALPHA):
    log_scores = {}
    past_users = list(item_r.get(i, {}).items())
    for y in R_values:
        log_s = math.log(calc_user_prior(u, y, a))
        for v, k in past_users:
            age    = global_max_ts - user_ts.get(v, {}).get(i, global_max_ts)
            w      = math.exp(-DECAY_LAMBDA * age)
            lik    = calc_ib_likelihood(u, y, v, k, a)
            log_s += w * (math.log(lik) if lik > 0 else -1e9)
        log_scores[y] = log_s
    return normalize_log_scores(log_scores)

def hybrid_temporal(u, i, a=ALPHA):
    ub_s = ub_score_temporal(u, i, a)
    ib_s = ib_score_temporal(u, i, a)
    n_Iu, n_Ui = len(user_r.get(u, {})), len(item_r.get(i, {}))
    w_ub, w_ib = 1.0 / (1 + n_Iu), 1.0 / (1 + n_Ui)
    hybrid = {y: max(ub_s[y], 1e-300) ** w_ub * max(ib_s[y], 1e-300) ** w_ib
              for y in R_values}
    total = sum(hybrid.values())
    return {y: v / total for y, v in hybrid.items()}, ub_s, ib_s

def prediction_confidence(scores_dict):
    probs     = list(scores_dict.values())
    entropy   = -sum(p * math.log(p + 1e-12) for p in probs)
    max_ent   = math.log(len(probs))
    return 1.0 - (entropy / max_ent) if max_ent > 0 else 1.0

# =====================================================================
# 3. TOP-K NEIGHBORHOOD SELECTION  (NEW)
# =====================================================================

def topk_items_for_ub(u, i, K=TOP_K):
    """Top-K past items of user u ranked by co-rater overlap with item i."""
    past = list(user_r.get(u, {}).items())
    if len(past) <= K:
        return past
    i_set = item_users_set.get(i, frozenset())
    return sorted(past,
                  key=lambda jk: len(i_set & item_users_set.get(jk[0], frozenset())),
                  reverse=True)[:K]

def topk_users_for_ib(u, i, K=TOP_K):
    """Top-K co-raters of item i ranked by item overlap with user u."""
    past = list(item_r.get(i, {}).items())
    if len(past) <= K:
        return past
    u_set = user_items_set.get(u, frozenset())
    return sorted(past,
                  key=lambda vk: len(u_set & user_items_set.get(vk[0], frozenset())),
                  reverse=True)[:K]

def confidence_maxprob(scores_dict):
    """Max-probability confidence: probability assigned to the predicted class.
    Range [0.2, 1.0] for 5 classes — works correctly with geometric hybrid mixing."""
    return max(scores_dict.values())

# =====================================================================
# 4. TOP-K + TEMPORAL SCORE FUNCTIONS  (NEW)
# =====================================================================

def ub_score_topk_temporal(u, i, a=ALPHA):
    log_scores = {}
    neighbors  = topk_items_for_ub(u, i)
    for y in R_values:
        log_s = math.log(calc_item_prior(i, y, a))
        for j, k in neighbors:
            w      = temporal_weight(u, j)
            lik    = calc_ub_likelihood(i, y, j, k, a)
            log_s += w * (math.log(lik) if lik > 0 else -1e9)
        log_scores[y] = log_s
    return normalize_log_scores(log_scores)

def ib_score_topk_temporal(u, i, a=ALPHA):
    log_scores = {}
    neighbors  = topk_users_for_ib(u, i)
    for y in R_values:
        log_s = math.log(calc_user_prior(u, y, a))
        for v, k in neighbors:
            age    = global_max_ts - user_ts.get(v, {}).get(i, global_max_ts)
            w      = math.exp(-DECAY_LAMBDA * age)
            lik    = calc_ib_likelihood(u, y, v, k, a)
            log_s += w * (math.log(lik) if lik > 0 else -1e9)
        log_scores[y] = log_s
    return normalize_log_scores(log_scores)

def hybrid_topk_temporal(u, i, a=ALPHA):
    ub_s = ub_score_topk_temporal(u, i, a)
    ib_s = ib_score_topk_temporal(u, i, a)
    n_Iu, n_Ui = len(user_r.get(u, {})), len(item_r.get(i, {}))
    w_ub, w_ib = 1.0 / (1 + n_Iu), 1.0 / (1 + n_Ui)
    hybrid = {y: max(ub_s[y], 1e-300) ** w_ub * max(ib_s[y], 1e-300) ** w_ib
              for y in R_values}
    total = sum(hybrid.values())
    return {y: v / total for y, v in hybrid.items()}, ub_s, ib_s

# =====================================================================
# 5. MAIN EVALUATION  (medium activity users, 1000 samples)
# =====================================================================
print("[2] Main evaluation (20-30 rating users, 1000 samples)...")

small_users = {u for u in user_r if 20 <= len(user_r[u]) <= 30}
eval_df     = test_df[test_df['user_id'].isin(small_users)].head(1000)

errors_ub, errors_ib, errors_h, errors_ht, errors_htk = [], [], [], [], []
actuals, preds_h, preds_ht, preds_htk = [], [], [], []
confidences_entropy, confidences_max   = [], []
latencies_h, latencies_ht, latencies_htk = [], [], []

for _, row in eval_df.iterrows():
    u, i, true_r = int(row['user_id']), int(row['item_id']), int(row['rating'])

    t0 = time.perf_counter()
    h_s,  ub_s,  ib_s  = hybrid_score(u, i)
    latencies_h.append(time.perf_counter() - t0)

    t1 = time.perf_counter()
    ht_s, ubt_s, ibt_s = hybrid_temporal(u, i)
    latencies_ht.append(time.perf_counter() - t1)

    t2 = time.perf_counter()
    htk_s, ubtk_s, ibtk_s = hybrid_topk_temporal(u, i)
    latencies_htk.append(time.perf_counter() - t2)

    p_ub  = predict_argmax(ub_s)
    p_ib  = predict_argmax(ib_s)
    p_h   = predict_argmax(h_s)
    p_ht  = predict_argmax(ht_s)
    p_htk = predict_argmax(htk_s)

    errors_ub.append(abs(true_r - p_ub))
    errors_ib.append(abs(true_r - p_ib))
    errors_h.append(abs(true_r - p_h))
    errors_ht.append(abs(true_r - p_ht))
    errors_htk.append(abs(true_r - p_htk))

    actuals.append(true_r)
    preds_h.append(p_h)
    preds_ht.append(p_ht)
    preds_htk.append(p_htk)

    confidences_entropy.append(prediction_confidence(ubt_s))
    confidences_max.append(confidence_maxprob(ubtk_s))

mae_ub  = np.mean(errors_ub)
mae_ib  = np.mean(errors_ib)
mae_h   = np.mean(errors_h)
mae_ht  = np.mean(errors_ht)
mae_htk = np.mean(errors_htk)

rmse_ub  = np.sqrt(np.mean(np.array(errors_ub)  ** 2))
rmse_ib  = np.sqrt(np.mean(np.array(errors_ib)  ** 2))
rmse_h   = np.sqrt(np.mean(np.array(errors_h)   ** 2))
rmse_ht  = np.sqrt(np.mean(np.array(errors_ht)  ** 2))
rmse_htk = np.sqrt(np.mean(np.array(errors_htk) ** 2))

print(f"  MAE  — UB:{mae_ub:.4f}  IB:{mae_ib:.4f}  Hybrid:{mae_h:.4f}  "
      f"Temporal:{mae_ht:.4f}  TopK:{mae_htk:.4f}")
print(f"  RMSE — UB:{rmse_ub:.4f}  IB:{rmse_ib:.4f}  Hybrid:{rmse_h:.4f}  "
      f"Temporal:{rmse_ht:.4f}  TopK:{rmse_htk:.4f}")

prec_h  = precision_score(actuals, preds_h,   average='macro', zero_division=0)
rec_h   = recall_score   (actuals, preds_h,   average='macro', zero_division=0)
f1_h    = f1_score       (actuals, preds_h,   average='macro', zero_division=0)
prec_ht = precision_score(actuals, preds_ht,  average='macro', zero_division=0)
rec_ht  = recall_score   (actuals, preds_ht,  average='macro', zero_division=0)
f1_ht   = f1_score       (actuals, preds_ht,  average='macro', zero_division=0)
prec_htk= precision_score(actuals, preds_htk, average='macro', zero_division=0)
rec_htk = recall_score   (actuals, preds_htk, average='macro', zero_division=0)
f1_htk  = f1_score       (actuals, preds_htk, average='macro', zero_division=0)

print(f"  Hybrid    — P:{prec_h:.4f}  R:{rec_h:.4f}  F1:{f1_h:.4f}")
print(f"  Temporal  — P:{prec_ht:.4f}  R:{rec_ht:.4f}  F1:{f1_ht:.4f}")
print(f"  TopK+Temp — P:{prec_htk:.4f}  R:{rec_htk:.4f}  F1:{f1_htk:.4f}")

act_bin  = [0 if a <= 3 else 1 for a in actuals]
pred_bin = [0 if p <= 3 else 1 for p in preds_htk]
cm_bin   = confusion_matrix(act_bin, pred_bin, labels=[0, 1])
cm_full  = confusion_matrix(actuals, preds_htk, labels=R_values)

# Confidence filtering with max-probability metric
CONF_THRESHOLD = 0.45
high_conf_mask = [c >= CONF_THRESHOLD for c in confidences_max]
coverage     = sum(high_conf_mask) / len(high_conf_mask)
idx_hc       = [m for m, b in enumerate(high_conf_mask) if b]
mae_filtered = float(np.mean(np.array(errors_htk)[idx_hc])) if idx_hc else 0.0
print(f"  Max-prob confidence @ {CONF_THRESHOLD}: coverage={coverage:.1%}  "
      f"filtered MAE={mae_filtered:.4f}\n")

# =====================================================================
# 6. USER GROUP EVALUATION  (NEW — Light / Medium / Heavy)
# =====================================================================
print("[3] User group evaluation (Light/Medium/Heavy, 300 samples each)...")

groups = {
    'Light\n(10–30)' : {u for u in user_r if 10  <= len(user_r[u]) <= 30},
    'Medium\n(31–100)': {u for u in user_r if 31  <= len(user_r[u]) <= 100},
    'Heavy\n(>100)'  : {u for u in user_r if len(user_r[u]) > 100},
}
GROUP_SAMPLES = 300

group_results = {}
for label, uset in groups.items():
    subset = test_df[test_df['user_id'].isin(uset)].head(GROUP_SAMPLES)
    e_ht, e_htk = [], []
    for _, row in subset.iterrows():
        u, i, true_r = int(row['user_id']), int(row['item_id']), int(row['rating'])
        ht_s,  *_  = hybrid_temporal(u, i)
        htk_s, *_  = hybrid_topk_temporal(u, i)
        e_ht.append(abs(true_r - predict_argmax(ht_s)))
        e_htk.append(abs(true_r - predict_argmax(htk_s)))

    arr_ht  = np.array(e_ht)
    arr_htk = np.array(e_htk)
    group_results[label] = {
        'mae_ht' : float(np.mean(arr_ht)),
        'mae_htk': float(np.mean(arr_htk)),
        'rmse_ht' : float(np.sqrt(np.mean(arr_ht  ** 2))),
        'rmse_htk': float(np.sqrt(np.mean(arr_htk ** 2))),
        'n': len(subset),
    }
    print(f"  {label.replace(chr(10),' ')} — n={len(subset)} | "
          f"MAE  Temporal:{group_results[label]['mae_ht']:.4f}  "
          f"TopK:{group_results[label]['mae_htk']:.4f} | "
          f"RMSE Temporal:{group_results[label]['rmse_ht']:.4f}  "
          f"TopK:{group_results[label]['rmse_htk']:.4f}")

# =====================================================================
# 7. ALPHA SENSITIVITY ANALYSIS  (NEW)
# =====================================================================
print("\n[4] Alpha sensitivity (UB + UB-Temporal, 200 samples)...")

alpha_values = [0.001, 0.01, 0.1, 0.5, 1.0]
alpha_subset = test_df[test_df['user_id'].isin(small_users)].head(200)

alpha_mae_ub  = []
alpha_mae_ubt = []

for a_val in alpha_values:
    e_ub, e_ubt = [], []
    for _, row in alpha_subset.iterrows():
        u, i, true_r = int(row['user_id']), int(row['item_id']), int(row['rating'])
        p_ub  = predict_argmax(ub_score(u, i, a=a_val))
        p_ubt = predict_argmax(ub_score_temporal(u, i, a=a_val))
        e_ub.append(abs(true_r - p_ub))
        e_ubt.append(abs(true_r - p_ubt))
    alpha_mae_ub.append(float(np.mean(e_ub)))
    alpha_mae_ubt.append(float(np.mean(e_ubt)))
    print(f"  alpha={a_val:.3f}  UB:{alpha_mae_ub[-1]:.4f}  UB-Temporal:{alpha_mae_ubt[-1]:.4f}")

# =====================================================================
# 8. SUSTAINABILITY: Computational Profiling  (same as midterm)
# =====================================================================
avg_lat_h   = np.mean(latencies_h)   * 1000
avg_lat_ht  = np.mean(latencies_ht)  * 1000
avg_lat_htk = np.mean(latencies_htk) * 1000
p95_lat_h   = np.percentile(latencies_h,   95) * 1000
p95_lat_ht  = np.percentile(latencies_ht,  95) * 1000
p95_lat_htk = np.percentile(latencies_htk, 95) * 1000

print(f"\n[5] Latency — avg ms (p95 ms):")
print(f"  Hybrid:      {avg_lat_h:.1f} ({p95_lat_h:.1f})")
print(f"  Temporal:    {avg_lat_ht:.1f} ({p95_lat_ht:.1f})")
print(f"  TopK+Temp:   {avg_lat_htk:.1f} ({p95_lat_htk:.1f})")

# =====================================================================
# 9. VISUALIZATION HELPERS
# =====================================================================
print("\n[6] Generating all visualisations...")

def styled_ax(ax, title, xlabel="", ylabel=""):
    ax.set_facecolor(BG)
    ax.set_title(title, fontweight='bold', color=BLUE, pad=8, fontsize=10)
    if xlabel: ax.set_xlabel(xlabel, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines[['top', 'right']].set_visible(False)

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

# =====================================================================
# FIGURE SECTION A: Tables 1–5  (same as midterm)
# =====================================================================
target_u     = test_df.iloc[0]['user_id']
target_i     = test_df.iloc[0]['item_id']
u_items      = list(user_r[target_u].keys())[:4]
other_items  = [x for x in item_r if x not in user_r[target_u] and x != target_i][:4]
selected_items = [target_i] + u_items + other_items
i_users      = [u for u in item_r[target_i] if u != target_u][:4]
selected_users = [target_u] + i_users

fig, ax = plt.subplots(figsize=(10, 3))
mat_data = []
for u in selected_users:
    row = [f"u_{u}"] + [user_r[u].get(i, '•') for i in selected_items]
    mat_data.append(row)
df1 = pd.DataFrame(mat_data, columns=[""] + [f"i_{x}" for x in selected_items])
draw_academic_table(ax, df1, "TABLE 1. Running example of the rating matrix (ML-1M).")
plt.tight_layout()
plt.savefig('outputs/table1_rating_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
ub_priors = [calc_item_prior(target_i, y) for y in R_values]
ib_priors = [calc_user_prior(target_u, y) for y in R_values]
df2 = pd.DataFrame([
    ["user-based"] + [f"{v:.6f}" for v in ub_priors],
    ["item-based"] + [f"{v:.6f}" for v in ib_priors]
], columns=[""] + R_values)
draw_academic_table(ax1, df2, f"TABLE 2. Prior probabilities (Item {target_i}, User {target_u}).")

t3_data = []
for j in u_items:
    row = [f"i_{j}"] + [f"{calc_ub_likelihood(target_i, y, j, user_r[target_u][j]):.6f}"
                         for y in R_values]
    t3_data.append(row)
df3 = pd.DataFrame(t3_data, columns=["j"] + R_values)
draw_academic_table(ax2, df3, f"TABLE 3. User-based likelihood for item {target_i}.")
plt.tight_layout()
plt.savefig('outputs/table2_3_priors_likelihood.png', dpi=150, bbox_inches='tight')
plt.close()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
t4_data = []
for v in i_users:
    row = [f"u_{v}"] + [f"{calc_ib_likelihood(target_u, y, v, item_r[target_i][v]):.8f}"
                         for y in R_values]
    t4_data.append(row)
df4 = pd.DataFrame(t4_data, columns=["v"] + R_values)
draw_academic_table(ax1, df4, f"TABLE 4. Item-based likelihood for user {target_u}.")

h_s,   ub_s,   ib_s   = hybrid_score(target_u, target_i)
ht_s,  ubt_s,  ibt_s  = hybrid_temporal(target_u, target_i)
htk_s, ubtk_s, ibtk_s = hybrid_topk_temporal(target_u, target_i)

def format_row(name, sd):
    mk = predict_argmax(sd)
    return [name] + [f"*{sd[y]:.5f}*" if y == mk else f"{sd[y]:.5f}" for y in R_values]

df5 = pd.DataFrame([
    format_row("UB",            ub_s),
    format_row("IB",            ib_s),
    format_row("Hybrid",        h_s),
    format_row("Temporal-UB",   ubt_s),
    format_row("Temporal-IB",   ibt_s),
    format_row("Hybrid-Temp",   ht_s),
    format_row("TopK-UB",       ubtk_s),
    format_row("TopK-IB",       ibtk_s),
    format_row("TopK+Temp [NEW]", htk_s),
], columns=["Method"] + R_values)
draw_academic_table(ax2, df5, "TABLE 5. Classification scores — all variants (* = predicted class).")
plt.tight_layout()
plt.savefig('outputs/table4_5_ib_scores.png', dpi=150, bbox_inches='tight')
plt.close()

# =====================================================================
# FIGURE 1: Confusion Matrices  (updated: now TopK+Temporal)
# =====================================================================
cm_norm     = cm_full.astype('float') / cm_full.sum(axis=1, keepdims=True)
cm_norm     = np.nan_to_num(cm_norm)
cm_bin_norm = cm_bin.astype('float') / cm_bin.sum(axis=1, keepdims=True)
cm_bin_norm = np.nan_to_num(cm_bin_norm)

fig, axes = plt.subplots(2, 2, figsize=(11, 9))
fig.suptitle("FIGURE 1. Confusion Matrices — TopK+Temporal Hybrid NBCF (ML-1M)",
             fontweight='bold', color=BLUE, fontsize=12, y=0.98)

sns.heatmap(cm_full, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=R_values, yticklabels=R_values, ax=axes[0, 0])
axes[0, 0].set_title("a) 5-Class — Raw Counts", fontsize=10)
axes[0, 0].set_ylabel("Actual"); axes[0, 0].set_xlabel("Predicted")

sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Blues', cbar=True,
            xticklabels=R_values, yticklabels=R_values, ax=axes[0, 1])
axes[0, 1].set_title("b) 5-Class — Normalised", fontsize=10)
axes[0, 1].set_ylabel("Actual"); axes[0, 1].set_xlabel("Predicted")

sns.heatmap(cm_bin, annot=True, fmt='d', cmap='Oranges', cbar=True,
            xticklabels=['Not Like', 'Like'], yticklabels=['Not Like', 'Like'],
            ax=axes[1, 0])
axes[1, 0].set_title("c) Binary — Raw Counts", fontsize=10)
axes[1, 0].set_ylabel("Actual"); axes[1, 0].set_xlabel("Predicted")

sns.heatmap(cm_bin_norm, annot=True, fmt='.1%', cmap='Oranges', cbar=True,
            xticklabels=['Not Like', 'Like'], yticklabels=['Not Like', 'Like'],
            ax=axes[1, 1])
axes[1, 1].set_title("d) Binary — Normalised", fontsize=10)
axes[1, 1].set_ylabel("Actual"); axes[1, 1].set_xlabel("Predicted")

plt.tight_layout()
plt.savefig('outputs/figure1_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()

# =====================================================================
# FIGURE 2: MAE + RMSE Comparison  (updated: 5 methods)
# =====================================================================
methods = ['UB', 'IB', 'Hybrid', 'Hybrid\n+Temporal', 'TopK\n+Temporal\n[NEW]']
maes    = [mae_ub, mae_ib, mae_h, mae_ht, mae_htk]
rmses   = [rmse_ub, rmse_ib, rmse_h, rmse_ht, rmse_htk]
colors  = [GRAY, GRAY, LIGHT_BLU, ORANGE, GREEN]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

bars1 = ax1.bar(methods, maes, color=colors, width=0.5, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars1, maes):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
             f"{val:.4f}", ha='center', fontsize=8, fontweight='bold')
ax1.set_ylim(0, max(maes) * 1.25)
styled_ax(ax1, "FIGURE 2a. MAE Comparison — All NBCF Variants", "Method", "MAE")

bars2 = ax2.bar(methods, rmses, color=colors, width=0.5, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars2, rmses):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
             f"{val:.4f}", ha='center', fontsize=8, fontweight='bold')
ax2.set_ylim(0, max(rmses) * 1.25)
styled_ax(ax2, "FIGURE 2b. RMSE Comparison — All NBCF Variants", "Method", "RMSE")

patches = [mpatches.Patch(color=LIGHT_BLU, label='Original Hybrid'),
           mpatches.Patch(color=ORANGE,    label='Hybrid + Temporal'),
           mpatches.Patch(color=GREEN,     label='TopK + Temporal [NEW]')]
ax1.legend(handles=patches, fontsize=8)
ax2.legend(handles=patches, fontsize=8)
plt.tight_layout()
plt.savefig('outputs/figure2_mae_rmse_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# =====================================================================
# FIGURE 3: Per-Class F1  (updated: 3 models)
# =====================================================================
f1_h_cls   = f1_score(actuals, preds_h,   average=None, labels=R_values, zero_division=0)
f1_ht_cls  = f1_score(actuals, preds_ht,  average=None, labels=R_values, zero_division=0)
f1_htk_cls = f1_score(actuals, preds_htk, average=None, labels=R_values, zero_division=0)

fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(R_values))
w = 0.25
ax.bar(x - w,   f1_h_cls,   w, label='Hybrid',           color=LIGHT_BLU, edgecolor='white')
ax.bar(x,       f1_ht_cls,  w, label='Hybrid+Temporal',   color=ORANGE,    edgecolor='white')
ax.bar(x + w,   f1_htk_cls, w, label='TopK+Temporal[NEW]',color=GREEN,     edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels([f"Rating {r}" for r in R_values])
styled_ax(ax, "FIGURE 3. Per-Class F1 Score by Rating Value", "Rating Class", "F1 Score")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('outputs/figure3_per_class_f1.png', dpi=150, bbox_inches='tight')
plt.close()

# =====================================================================
# FIGURE 4: Confidence Analysis  (updated: max-prob metric)
# =====================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

ax1.hist(confidences_max, bins=20, color=LIGHT_BLU, edgecolor='white', alpha=0.85)
ax1.axvline(CONF_THRESHOLD, color=ORANGE, linestyle='--', lw=2,
            label=f'Threshold={CONF_THRESHOLD}')
styled_ax(ax1, "FIGURE 4a. Max-Prob Confidence Distribution\n(TopK-UB Temporal)",
          "Confidence Score", "Count")
ax1.legend(fontsize=9)

buckets    = np.linspace(0.2, 1.0, 11)
b_mae, b_cov = [], []
for lo, hi in zip(buckets[:-1], buckets[1:]):
    mask = [lo <= c < hi for c in confidences_max]
    errs = [e for e, m in zip(errors_htk, mask) if m]
    b_mae.append(np.mean(errs) if errs else np.nan)
    b_cov.append(sum(mask))

mids  = (buckets[:-1] + buckets[1:]) / 2
valid = ~np.isnan(b_mae)
ax2.plot(mids[valid], np.array(b_mae)[valid], 'o-', color=ORANGE, lw=2, ms=6)
ax2.set_xlim(0.2, 1.0)
styled_ax(ax2, "FIGURE 4b. MAE vs. Prediction Confidence Bucket",
          "Confidence Bucket", "Mean MAE")
plt.tight_layout()
plt.savefig('outputs/figure4_confidence.png', dpi=150, bbox_inches='tight')
plt.close()

# =====================================================================
# FIGURE 5: Rating Distribution  (same)
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

act_cnt  = [actuals.count(r)   for r in R_values]
pred_h_c = [preds_h.count(r)   for r in R_values]
pred_ht_c= [preds_ht.count(r)  for r in R_values]
pred_htk_c=[preds_htk.count(r) for r in R_values]

x = np.arange(len(R_values))
w = 0.22
axes[0].bar(x - 1.5*w, act_cnt,   w, label='Actual',           color=GRAY,      edgecolor='white')
axes[0].bar(x - 0.5*w, pred_h_c,  w, label='Hybrid',           color=LIGHT_BLU, edgecolor='white')
axes[0].bar(x + 0.5*w, pred_ht_c, w, label='Temporal',         color=ORANGE,    edgecolor='white')
axes[0].bar(x + 1.5*w, pred_htk_c,w, label='TopK+Temp [NEW]',  color=GREEN,     edgecolor='white')
axes[0].set_xticks(x); axes[0].set_xticklabels(R_values)
styled_ax(axes[0], "FIGURE 5a. Predicted vs Actual Rating Distribution", "Rating", "Count")
axes[0].legend(fontsize=8)

bias_h   = [np.mean([e for a, e in zip(actuals, errors_h)   if a == r]) for r in R_values]
bias_ht  = [np.mean([e for a, e in zip(actuals, errors_ht)  if a == r]) for r in R_values]
bias_htk = [np.mean([e for a, e in zip(actuals, errors_htk) if a == r]) for r in R_values]
axes[1].plot(R_values, bias_h,   'o-', color=LIGHT_BLU, lw=2, ms=7, label='Hybrid')
axes[1].plot(R_values, bias_ht,  's-', color=ORANGE,    lw=2, ms=7, label='Temporal')
axes[1].plot(R_values, bias_htk, 'D-', color=GREEN,     lw=2, ms=7, label='TopK+Temp')
styled_ax(axes[1], "FIGURE 5b. MAE per Rating Class", "Actual Rating", "MAE")
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig('outputs/figure5_rating_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# =====================================================================
# FIGURE 6: Sustainability — Latency  (updated: 3 models)
# =====================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

ax1.hist([l*1000 for l in latencies_h],   bins=25, alpha=0.6, color=LIGHT_BLU, label='Hybrid',     edgecolor='white')
ax1.hist([l*1000 for l in latencies_ht],  bins=25, alpha=0.6, color=ORANGE,    label='Temporal',   edgecolor='white')
ax1.hist([l*1000 for l in latencies_htk], bins=25, alpha=0.6, color=GREEN,     label='TopK+Temp',  edgecolor='white')
ax1.axvline(avg_lat_h,   color=BLUE,    linestyle='--', lw=1.5, label=f'avg H={avg_lat_h:.1f}ms')
ax1.axvline(avg_lat_ht,  color='#b84c00', linestyle='--', lw=1.5, label=f'avg T={avg_lat_ht:.1f}ms')
ax1.axvline(avg_lat_htk, color='#1a6b3a', linestyle='--', lw=1.5, label=f'avg K={avg_lat_htk:.1f}ms')
styled_ax(ax1, "FIGURE 6a. Prediction Latency Distribution", "Latency (ms)", "Count")
ax1.legend(fontsize=7)

depth_bins = [10, 20, 30, 50, 80, 120]
tput_ht, tput_htk = [], []
for lo, hi in zip(depth_bins[:-1], depth_bins[1:]):
    mask   = [lo <= len(user_r.get(int(u), {})) < hi for u in eval_df['user_id']]
    lats_ht  = [l for l, m in zip(latencies_ht,  mask) if m]
    lats_htk = [l for l, m in zip(latencies_htk, mask) if m]
    tput_ht.append(1 / np.mean(lats_ht)  if lats_ht  else 0)
    tput_htk.append(1 / np.mean(lats_htk) if lats_htk else 0)

mids = [(lo + hi) / 2 for lo, hi in zip(depth_bins[:-1], depth_bins[1:])]
ax2.plot(mids, tput_ht,  's-', color=ORANGE, lw=2, ms=7, label='Temporal')
ax2.plot(mids, tput_htk, 'D-', color=GREEN,  lw=2, ms=7, label='TopK+Temp')
styled_ax(ax2, "FIGURE 6b. Throughput vs. User History Depth", "Avg History Size", "Pred/sec")
ax2.legend(fontsize=9)
plt.tight_layout()
plt.savefig('outputs/figure6_sustainability.png', dpi=150, bbox_inches='tight')
plt.close()

# =====================================================================
# FIGURE 7: User Group Performance — NEW
# =====================================================================
group_labels = list(group_results.keys())
mae_ht_g    = [group_results[g]['mae_ht']  for g in group_labels]
mae_htk_g   = [group_results[g]['mae_htk'] for g in group_labels]
rmse_ht_g   = [group_results[g]['rmse_ht']  for g in group_labels]
rmse_htk_g  = [group_results[g]['rmse_htk'] for g in group_labels]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
x = np.arange(len(group_labels))
w = 0.35

bars_ht  = ax1.bar(x - w/2, mae_ht_g,  w, color=ORANGE, edgecolor='white', label='Hybrid+Temporal')
bars_htk = ax1.bar(x + w/2, mae_htk_g, w, color=GREEN,  edgecolor='white', label='TopK+Temporal [NEW]')
for bar, val in zip(bars_ht,  mae_ht_g):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f"{val:.3f}", ha='center', fontsize=8)
for bar, val in zip(bars_htk, mae_htk_g):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f"{val:.3f}", ha='center', fontsize=8)
ax1.set_xticks(x); ax1.set_xticklabels(group_labels, fontsize=9)
styled_ax(ax1, "FIGURE 7a. MAE by User Activity Group", "User Group", "MAE")
ax1.legend(fontsize=9)

bars_ht2  = ax2.bar(x - w/2, rmse_ht_g,  w, color=ORANGE, edgecolor='white', label='Hybrid+Temporal')
bars_htk2 = ax2.bar(x + w/2, rmse_htk_g, w, color=GREEN,  edgecolor='white', label='TopK+Temporal [NEW]')
for bar, val in zip(bars_ht2,  rmse_ht_g):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f"{val:.3f}", ha='center', fontsize=8)
for bar, val in zip(bars_htk2, rmse_htk_g):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f"{val:.3f}", ha='center', fontsize=8)
ax2.set_xticks(x); ax2.set_xticklabels(group_labels, fontsize=9)
styled_ax(ax2, "FIGURE 7b. RMSE by User Activity Group", "User Group", "RMSE")
ax2.legend(fontsize=9)
plt.tight_layout()
plt.savefig('outputs/figure7_user_groups.png', dpi=150, bbox_inches='tight')
plt.close()

# =====================================================================
# FIGURE 8: Confidence Threshold Sweep — NEW
# =====================================================================
thresholds = np.arange(0.20, 0.96, 0.05)
cov_vals, fmae_vals = [], []
for thr in thresholds:
    mask = [c >= thr for c in confidences_max]
    cov  = sum(mask) / len(mask)
    errs = [e for e, m in zip(errors_htk, mask) if m]
    fmae = float(np.mean(errs)) if errs else np.nan
    cov_vals.append(cov * 100)
    fmae_vals.append(fmae)

valid = ~np.isnan(fmae_vals)
thresholds_v = thresholds[valid]
cov_v    = np.array(cov_vals)[valid]
fmae_v   = np.array(fmae_vals)[valid]

fig, ax1 = plt.subplots(figsize=(9, 4))
color_cov  = BLUE
color_mae  = ORANGE
ax2 = ax1.twinx()

ax1.plot(thresholds_v, cov_v,  'o-', color=color_cov, lw=2, ms=6, label='Coverage (%)')
ax2.plot(thresholds_v, fmae_v, 's--', color=color_mae, lw=2, ms=6, label='Filtered MAE')

ax1.set_xlabel("Confidence Threshold", fontsize=9)
ax1.set_ylabel("Coverage (%)", color=color_cov, fontsize=9)
ax2.set_ylabel("Filtered MAE",  color=color_mae, fontsize=9)
ax1.tick_params(axis='y', labelcolor=color_cov)
ax2.tick_params(axis='y', labelcolor=color_mae)
ax1.set_facecolor(BG)
ax1.set_title("FIGURE 8. Coverage–MAE Trade-off vs. Confidence Threshold\n"
              "(Max-Probability, TopK+Temporal UB scores)",
              fontweight='bold', color=BLUE, pad=8, fontsize=10)
ax1.spines[['top', 'right']].set_visible(False)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
plt.tight_layout()
plt.savefig('outputs/figure8_confidence_sweep.png', dpi=150, bbox_inches='tight')
plt.close()

# =====================================================================
# FIGURE 9: Alpha Sensitivity — NEW
# =====================================================================
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(alpha_values, alpha_mae_ub,  'o-', color=LIGHT_BLU, lw=2, ms=7, label='UB (no temporal)')
ax.plot(alpha_values, alpha_mae_ubt, 's-', color=ORANGE,    lw=2, ms=7, label='UB + Temporal')
ax.set_xscale('log')
ax.axvline(ALPHA, color=GRAY, linestyle='--', lw=1.5, label=f'Default α={ALPHA}')
styled_ax(ax, "FIGURE 9. MAE vs. Laplace Smoothing Parameter (α)",
          "Alpha (log scale)", "MAE")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('outputs/figure9_alpha_sensitivity.png', dpi=150, bbox_inches='tight')
plt.close()

# =====================================================================
# TABLE 6: User Group Breakdown — NEW
# =====================================================================
fig, ax = plt.subplots(figsize=(11, 3))
df6 = pd.DataFrame([
    [lbl.replace('\n', ' '),
     str(group_results[lbl]['n']),
     f"{group_results[lbl]['mae_ht']:.5f}",
     f"{group_results[lbl]['mae_htk']:.5f}",
     f"{group_results[lbl]['rmse_ht']:.5f}",
     f"{group_results[lbl]['rmse_htk']:.5f}"]
    for lbl in group_labels
], columns=["User Group", "n", "MAE Temporal", "MAE TopK [NEW]",
            "RMSE Temporal", "RMSE TopK [NEW]"])
draw_academic_table(ax, df6, "TABLE 6. Performance by User Activity Group — TopK vs. Temporal.")
plt.tight_layout()
plt.savefig('outputs/table6_user_groups.png', dpi=150, bbox_inches='tight')
plt.close()

# =====================================================================
# TABLE 7: Full Metrics Summary — updated (+ RMSE, + TopK model)
# =====================================================================
fig, ax = plt.subplots(figsize=(13, 4))
df7 = pd.DataFrame({
    "Method": [
        "NBCF (user-based)",
        "NBCF (item-based)",
        "NBCF (hybrid)",
        "NBCF hybrid + temporal decay",
        "NBCF TopK + temporal decay [NEW]",
    ],
    "MAE":        [f"{mae_ub:.5f}", f"{mae_ib:.5f}", f"{mae_h:.5f}",   f"{mae_ht:.5f}",  f"{mae_htk:.5f}"],
    "RMSE":       [f"{rmse_ub:.5f}",f"{rmse_ib:.5f}",f"{rmse_h:.5f}",  f"{rmse_ht:.5f}", f"{rmse_htk:.5f}"],
    "Precision":  ["—", "—", f"{prec_h:.5f}",  f"{prec_ht:.5f}",  f"{prec_htk:.5f}"],
    "Recall":     ["—", "—", f"{rec_h:.5f}",   f"{rec_ht:.5f}",   f"{rec_htk:.5f}"],
    "F1 (macro)": ["—", "—", f"{f1_h:.5f}",    f"{f1_ht:.5f}",    f"{f1_htk:.5f}"],
    "Coverage":   ["—", "—", "—", "—",          f"{coverage:.1%}"],
})
draw_academic_table(ax, df7, "TABLE 7. Comprehensive evaluation metrics for all NBCF variants.")
plt.tight_layout()
plt.savefig('outputs/table7_full_metrics.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "=" * 72)
print("  All outputs saved to outputs/")
print("  Figures : 9 (figure1–figure9)")
print("  Tables  : 7 (table1–table7)")
print("=" * 72)
