"""
03_model1_late.py   Model 1: "will this order be late: yes or no".

Run:  python 03_model1_late.py        (needs data/olist_order_level.csv from step 01)
Writes: models/olist_late_pipeline.joblib, models/feature_columns.json,
        outputs/model1_report.txt, outputs/tables/*.csv

Everyday picture: a weather forecaster says "rain tomorrow: yes/no". The honest
test is NOT "how often is the forecaster right" (if it rains 8 days in 100,
always saying "no rain" is right 92 times). The honest test is "of the days it
really rained, how many did you warn about, and how many useless warnings did you send".
"""
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parent
(BASE / "models").mkdir(exist_ok=True)
(BASE / "outputs" / "tables").mkdir(parents=True, exist_ok=True)
SEED = 42
EXPECTED_ORDERS = 95_128

report = []
def say(m=""):
    print(m); report.append(str(m))
def head(t):
    say("\n" + "=" * 72 + "\n" + t + "\n" + "=" * 72)

# ------------------------------------------------------------------ data
orders = pd.read_csv(BASE / "data" / "olist_order_level.csv")
if not orders["order_id"].is_unique or len(orders) != EXPECTED_ORDERS:
    raise SystemExit("STOPPED: order table is not 95,128 unique orders. Re-run 01_build_order_table.py")

CAT = ["customer_state", "purchase_dow"]
LOGNUM = ["approval_hours", "n_items", "price_total", "freight_total", "weight_total_g", "volume_total_cm3"]
PLAIN = ["est_window_days", "n_sellers", "any_cross_state"]
FEATURES = CAT + LOGNUM + PLAIN
FORBIDDEN = {"order_delivered_customer_date", "order_delivered_carrier_date", "seller_state",
             "payment_type", "payment_value", "payment_installments", "is_late"}
assert not FORBIDDEN & set(FEATURES), "a forbidden column is in the feature list"

X = orders[FEATURES].copy()
X["purchase_dow"] = X["purchase_dow"].astype(str)
y = orders["is_late"].values

# ------------------------------------------------------------------ ONE preprocessing pipeline (saved inside the model with joblib)
def make_pre(cat=None):
    cat = CAT if cat is None else cat
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
        ("log", Pipeline([("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                          ("scale", StandardScaler())]), LOGNUM),
        ("plain", StandardScaler(), PLAIN),
    ])

def make_lr():
    return Pipeline([("pre", make_pre()), ("model", LogisticRegression(max_iter=1000, C=1.0))])

def make_tree():
    return Pipeline([("pre", make_pre()),
                     ("model", HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=200,
                                                              random_state=SEED))])

# ------------------------------------------------------------------ split by order, stratified on late
head("1. SPLIT")
idx = np.arange(len(orders))
tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y[tr], y[te]
assert set(orders["order_id"].iloc[tr]).isdisjoint(set(orders["order_id"].iloc[te])), "an order is in both train and test"
say(f"train: {len(tr):,} orders, late {ytr.sum():,} ({ytr.mean():.2%})")
say(f"test : {len(te):,} orders, late {yte.sum():,} ({yte.mean():.2%})   <- natural late rate kept, not balanced")
say("check passed: no order_id appears in both train and test (one row per order, so this is guaranteed)")

# ------------------------------------------------------------------ helpers
def top_share(prob, truth, share=0.10):
    k = math.ceil(share * len(truth))
    order = np.argsort(-prob, kind="stable")[:k]
    caught = int(truth[order].sum())
    return k, caught, caught / truth.sum(), caught / k

def threshold_by_oof(model_factory, Xtr, ytr):
    """Pick the yes/no cut-off using only TRAIN data (5-fold out-of-sample predictions), maximising F1 for 'late'."""
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof = cross_val_predict(model_factory(), Xtr, ytr, cv=skf, method="predict_proba")[:, 1]
    grid = np.linspace(0.02, 0.60, 59)
    f1s = [f1_score(ytr, oof >= t) for t in grid]
    return float(grid[int(np.argmax(f1s))]), float(roc_auc_score(ytr, oof))

def evaluate(name, pipe, thr):
    p = pipe.predict_proba(Xte)[:, 1]
    flag = p >= thr
    tn, fp, fn, tp = confusion_matrix(yte, flag).ravel()
    k, caught10, cap10, prec10 = top_share(p, yte)
    res = dict(name=name, thr=thr, tp=int(tp), fn=int(fn), fp=int(fp), tn=int(tn),
               flagged=int(flag.sum()), precision=tp / max(tp + fp, 1), recall=tp / (tp + fn),
               auc=roc_auc_score(yte, p), ap=average_precision_score(yte, p),
               k10=k, caught10=caught10, cap10=cap10, prec10=prec10, p=p)
    return res

def bootstrap_ci(p, truth, reps=500):
    rng = np.random.default_rng(SEED)
    caps, aucs = [], []
    n = len(truth)
    for _ in range(reps):
        i = rng.integers(0, n, n)
        if truth[i].sum() == 0:
            continue
        caps.append(top_share(p[i], truth[i])[2]); aucs.append(roc_auc_score(truth[i], p[i]))
    return np.percentile(caps, [2.5, 97.5]), np.percentile(aucs, [2.5, 97.5])

# ------------------------------------------------------------------ baseline
head("2. BASELINE: 'never late'")
n_te, n_late_te = len(yte), int(yte.sum())
say(f"say 'not late' for every test order -> accuracy {1 - yte.mean():.2%}, real lates caught 0 of {n_late_te:,}")
say(f"flag a RANDOM 10% of orders -> catches about 10% of real lates (~{0.10 * n_late_te:.0f} of {n_late_te:,})")

# ------------------------------------------------------------------ logistic regression
head("3. LOGISTIC REGRESSION (main model)")
thr_lr, cv_auc_lr = threshold_by_oof(make_lr, Xtr, ytr)
say(f"yes/no cut-off chosen on TRAIN data only: flag 'late' when predicted chance >= {thr_lr:.2f}")
say(f"5-fold AUC on train (out-of-sample): {cv_auc_lr:.3f}")
lr = make_lr().fit(Xtr, ytr)
R = {"LR": evaluate("Logistic regression", lr, thr_lr)}

def show(r):
    say(f"\n[{r['name']}]  cut-off {r['thr']:.2f}, on {n_te:,} test orders ({n_late_te:,} really late)")
    say(f"  real lates caught (true alarms)   : {r['tp']:,} of {n_late_te:,}  = recall {r['recall']:.1%}")
    say(f"  real lates missed                 : {r['fn']:,}")
    say(f"  false alarms (flagged, not late)  : {r['fp']:,}")
    say(f"  orders flagged in total           : {r['flagged']:,} ({r['flagged'] / n_te:.1%} of test)")
    say(f"  precision for late                : {r['precision']:.1%}  (of every 100 flagged, {r['precision'] * 100:.0f} are really late; random flagging gives {yte.mean() * 100:.1f})")
    say(f"  accuracy                          : {(r['tp'] + r['tn']) / n_te:.2%}  (never-late gets {1 - yte.mean():.2%}; accuracy is NOT the score here)")
    say(f"  riskiest 10% ({r['k10']:,} orders)      : catches {r['caught10']:,} of {n_late_te:,} real lates = {r['cap10']:.1%}  (random would give ~10%); {r['prec10']:.1%} of them really late")
    say(f"  AUC {r['auc']:.3f}  (0.5 = coin flip, 1.0 = perfect) | average precision {r['ap']:.3f}  (random = {yte.mean():.3f})")
show(R["LR"])

# the 0.5 default, to show why a cut-off had to be chosen
p_lr = R["LR"]["p"]
f50 = p_lr >= 0.5
say(f"\nfor comparison, the default 0.50 cut-off flags {int(f50.sum())} orders and catches {int((f50 & (yte == 1)).sum())} real lates")

ci_cap, ci_auc = bootstrap_ci(p_lr, yte)
say(f"give-or-take (95% range from re-sampling the test set 500 times): riskiest-10% catch {ci_cap[0]:.1%} to {ci_cap[1]:.1%}; AUC {ci_auc[0]:.3f} to {ci_auc[1]:.3f}")

# ------------------------------------------------------------------ tree model, only if it adds clear value
head("4. TREE MODEL (HistGradientBoosting) vs LOGISTIC: does it add CLEAR value?")
RULE_CAP, RULE_AUC = 0.05, 0.02
say(f"rule fixed BEFORE looking: tree is worth keeping only if it beats logistic by >= {RULE_CAP:.0%} points on riskiest-10% catch AND >= {RULE_AUC} AUC")
thr_tr, cv_auc_tr = threshold_by_oof(make_tree, Xtr, ytr)
tree = make_tree().fit(Xtr, ytr)
R["TREE"] = evaluate("Tree (HistGradientBoosting, fixed settings, not tuned)", tree, thr_tr)
show(R["TREE"])
d_cap = R["TREE"]["cap10"] - R["LR"]["cap10"]; d_auc = R["TREE"]["auc"] - R["LR"]["auc"]
tree_wins = d_cap >= RULE_CAP and d_auc >= RULE_AUC
say(f"\ndifference tree - logistic: riskiest-10% catch {d_cap:+.1%} points, AUC {d_auc:+.3f}")
say("VERDICT: " + ("tree adds clear value" if tree_wins else "tree does NOT add clear value; logistic regression stays the model"))

# ------------------------------------------------------------------ permutation importance on test
head("5. PERMUTATION IMPORTANCE (test set, logistic regression)")
say("Shuffle one column at random and see how much the score (AUC) drops. Big drop = the model leaned on that column.")
pi = permutation_importance(lr, Xte, yte, scoring="roc_auc", n_repeats=10, random_state=SEED, n_jobs=1)
imp = pd.DataFrame({"feature": FEATURES, "auc_drop": pi.importances_mean, "spread": pi.importances_std}).sort_values("auc_drop", ascending=False)
say(imp.round(4).to_string(index=False))
imp.to_csv(BASE / "outputs" / "tables" / "permutation_importance_logreg.csv", index=False)
if tree_wins:
    pit = permutation_importance(tree, Xte, yte, scoring="roc_auc", n_repeats=10, random_state=SEED, n_jobs=1)
    say("\ntree model:")
    say(pd.DataFrame({"feature": FEATURES, "auc_drop": pit.importances_mean}).sort_values("auc_drop", ascending=False).round(4).to_string(index=False))

# ------------------------------------------------------------------ extra honesty check: train on the past, test on the newest orders
head("6. EXTRA CHECK: train on older orders, test on the NEWEST 20% (real-life direction)")
srt = orders.sort_values("order_purchase_timestamp").index.values
cut = int(0.8 * len(srt))
a, b = srt[:cut], srt[cut:]
lr_t = make_lr().fit(X.iloc[a], y[a])
pt = lr_t.predict_proba(X.iloc[b])[:, 1]
k, c10, cap, pr = top_share(pt, y[b])
say(f"train on orders purchased up to {orders.loc[a, 'order_purchase_timestamp'].max()[:10]}, "
    f"test on {len(b):,} newer orders (late rate {y[b].mean():.2%}; train late rate {y[a].mean():.2%})")
say(f"AUC {roc_auc_score(y[b], pt):.3f} | riskiest 10% catches {cap:.1%} of real lates")

# ------------------------------------------------------------------ extra: what if the model is told the purchase month?
head("6b. EXTRA CHECK: give the model the purchase MONTH (NOT used in the saved model)")
say("The month is known at approval, but it improves the score only on the random split, so it is not in the saved model.")
Xm = X.copy(); Xm["purchase_month"] = orders["purchase_month"].values
lr_m = Pipeline([("pre", make_pre(CAT + ["purchase_month"])), ("model", LogisticRegression(max_iter=1000))])
lr_m.fit(Xm.iloc[tr], ytr)
pm = lr_m.predict_proba(Xm.iloc[te])[:, 1]
_, _, cap_m, _ = top_share(pm, yte)
say(f"random split WITH month: AUC {roc_auc_score(yte, pm):.3f} (without: {R['LR']['auc']:.3f}); riskiest 10% catches {cap_m:.1%} (without: {R['LR']['cap10']:.1%})")
lr_m2 = Pipeline([("pre", make_pre(CAT + ["purchase_month"])), ("model", LogisticRegression(max_iter=1000))]).fit(Xm.iloc[a], y[a])
pm2 = lr_m2.predict_proba(Xm.iloc[b])[:, 1]
say(f"older->newest WITH month: AUC {roc_auc_score(y[b], pm2):.3f} (without: {roc_auc_score(y[b], pt):.3f}); riskiest 10% catches {top_share(pm2, y[b])[2]:.1%} (without: {top_share(pt, y[b])[2]:.1%})")
say("(newest months are unseen categories, so the month adds nothing there. The jump, if any, only shows on the random split.)")
AUC_M = roc_auc_score(yte, pm); CAP_M = cap_m
AUC_M2 = roc_auc_score(y[b], pm2)

# ------------------------------------------------------------------ plain late-rate tables (all 95,128 orders)
head("7. PLAIN LATE-RATE TABLES (all 95,128 orders; counts shown; descriptive, not model output)")
say(f"overall: {orders['is_late'].sum():,} late of {len(orders):,} = {orders['is_late'].mean():.2%}")
def rate_table(col_or_series, name, order=None, min_n=100):
    s = orders["is_late"].groupby(col_or_series, observed=True).agg(orders="size", late="sum")
    s["late_rate_%"] = (100 * s["late"] / s["orders"]).round(2)
    s["note"] = np.where(s["orders"] < min_n, "few orders: do not trust", "")
    if order is not None:
        s = s.reindex(order)
    else:
        s = s.sort_values("late_rate_%", ascending=False)
    say(f"\n--- by {name} ---"); say(s.to_string())
    s.to_csv(BASE / "outputs" / "tables" / f"late_rate_by_{name.replace(' ', '_')}.csv")

rate_table(orders["customer_state"], "customer state")

wb = pd.cut(orders["est_window_days"], [0, 15, 20, 25, 30, 40, 1000], right=False,
            labels=["under 15 days", "15-19.9", "20-24.9", "25-29.9", "30-39.9", "40+ days"])
rate_table(wb, "estimated window band", order=list(wb.cat.categories))

fb = pd.cut(orders["freight_total"], [-0.01, 0, 10, 15, 20, 30, 50, 1e9], right=True,
            labels=["R$0 (free)", "0.01-10", "10.01-15", "15.01-20", "20.01-30", "30.01-50", "over 50"])
rate_table(fb, "freight band", order=list(fb.cat.categories))

sc = orders["n_sellers"].clip(upper=3).map({1: "1 seller", 2: "2 sellers", 3: "3+ sellers"})
rate_table(sc, "number of sellers", order=["1 seller", "2 sellers", "3+ sellers"])

cs = orders["any_cross_state"].map({0: "all items from customer's state", 1: "some item from another state"})
rate_table(cs, "cross-state", order=["all items from customer's state", "some item from another state"])

rate_table(orders["purchase_month"], "purchase month (extra)", order=sorted(orders["purchase_month"].unique()), min_n=300)

# ------------------------------------------------------------------ save the pipeline, prove it reloads identically
head("8. SAVING (one pipeline = preprocessing + model, saved together)")
final_name = "TREE" if tree_wins else "LR"
final = tree if tree_wins else lr
final_thr = R[final_name]["thr"]
joblib.dump({"pipeline": final, "threshold": final_thr, "features": FEATURES, "model": R[final_name]["name"]},
            BASE / "models" / "olist_late_pipeline.joblib")
(BASE / "models" / "feature_columns.json").write_text(json.dumps({"features": FEATURES, "categorical": CAT}, indent=2))
back = joblib.load(BASE / "models" / "olist_late_pipeline.joblib")
assert np.allclose(back["pipeline"].predict_proba(Xte)[:, 1], R[final_name]["p"]), "reloaded model gives different answers"
say("check passed: the reloaded file gives exactly the same predictions as the model in memory")
enc = lr.named_steps["pre"].named_transformers_["cat"]
say(f"state codes live INSIDE the pipeline (one-hot encoder knows {len(enc.categories_[0])} states); "
    f"there is no separate state encoder that could drift")
unk = X.iloc[:3].copy(); unk.loc[unk.index[0], "customer_state"] = "ZZ"
assert np.isfinite(back["pipeline"].predict_proba(unk)).all()
say("check passed: an unknown state code does not crash the pipeline")
say(f"saved model: {R[final_name]['name']}, cut-off {final_thr:.2f}")
