"""
Model 5 (rebuild) - customer payment-behaviour segments, Olist cleaned file.

HOW TO RUN (from the repository root):
    python Model5\model5_payment_behaviour.py
Reads : Model5/data/Brazilian-Ecommerce-Cleaned.xlsx
Writes: Model5/outputs/model5/*.png, *.csv, README_module5_draft.md
        (the draft file is regenerated on every run and is not published)

UNITS (used in every print and chart):
  row      = one line of the file (an item x payment combination), 113,390
  item     = one product line of an order, 108,640
  payment  = one payment of an order (order_id + payment_sequential), 99,329
  order    = one distinct order_id, 95,128
  customer = one distinct customer_unique_id, 92,081
No setting was changed to improve a score.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# ----------------------------------------------------------------- settings
BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "Brazilian-Ecommerce-Cleaned.xlsx"
OUT = BASE / "outputs" / "model5"
SEED = 42                 # same seed as the earlier version
K_RANGE = range(2, 11)    # same k range as the earlier version
SAMPLE_N = 10_000         # sample used only to choose k (same size as earlier version)
STABILITY_SEEDS = [0, 1, 2, 3, 4]   # only to CHECK stability, never to pick a result
FULL_SILHOUETTE = False   # True = silhouette on ALL customers (like the earlier version; slow, several minutes)
SIL_SAMPLE = 20_000       # otherwise: silhouette on this many customers, drawn without repeats, seed 42

# Numbers printed by the earlier version (used only to confirm my replay of it)
OLD_NB = {
    "sizes_k2": [16467, 78661],
    "sil_sample": {2: 0.5923, 3: 0.5185, 4: 0.5184, 5: 0.5319, 6: 0.5235,
                   7: 0.3921, 8: 0.3974, 9: 0.3997, 10: 0.3900},
    "sil_final": 0.5833,
    "corr_inst_value": 0.2743,
    "mean_spend": [537.32, 135.80],
}


def head(t):
    print("\n" + "=" * 8 + " " + t + " " + "=" * 8)


def flag(ok):
    return "OK" if ok else "DIFFERENT"


def sil_final(X_, labels_):
    if FULL_SILHOUETTE:
        return silhouette_score(X_, labels_)
    return silhouette_score(X_, labels_, sample_size=SIL_SAMPLE, random_state=SEED)


OUT.mkdir(parents=True, exist_ok=True)
print("Python", sys.version.split()[0], "| pandas", pd.__version__, "| numpy", np.__version__,
      "| scikit-learn", sklearn.__version__, "| matplotlib", matplotlib.__version__)

# ----------------------------------------------------------------- load
t0 = time.time()
cols = ["order_id", "order_item_id", "customer_id", "customer_unique_id",
        "payment_sequential", "payment_type", "payment_installments",
        "price", "freight_value", "payment_value"]
df = pd.read_excel(DATA, usecols=cols)
print(f"loaded {len(df):,} rows in {time.time() - t0:.0f}s")

# ----------------------------------------------------------------- grain asserts
head("A. GRAIN CHECKS (the script stops if the file differs from the expected counts)")
items = df.drop_duplicates(["order_id", "order_item_id"])
pays = df.drop_duplicates(["order_id", "payment_sequential"])
n_orders = df["order_id"].nunique()
n_cust = df["customer_unique_id"].nunique()

assert len(df) == 113_390, "row count changed"
assert n_orders == 95_128, "distinct order count changed"
assert df["customer_id"].nunique() == 95_128
assert n_cust == 92_081, "distinct customer_unique_id changed"
assert len(items) == 108_640, "item count changed"
assert len(pays) == 99_329, "payment count changed"
assert df.groupby("order_id")["customer_unique_id"].nunique().max() == 1, \
    "an order has 2 customers"
assert (df.groupby(["order_id", "payment_sequential"])[
    ["payment_value", "payment_type", "payment_installments"]].nunique() == 1).all().all(), \
    "payment columns differ inside one payment"
assert (df.groupby(["order_id", "order_item_id"])[["price", "freight_value"]]
        .nunique() == 1).all().all(), "price/freight differ inside one item"
assert df[cols].isna().sum().sum() == 0, "missing values in used columns"
print(f"rows {len(df):,} | items {len(items):,} | payments {len(pays):,} | "
      f"orders {n_orders:,} | customers {n_cust:,}")

paid_rows = df["payment_value"].sum()
paid_once = pays["payment_value"].sum()
assert round(paid_once, 2) == 15_225_816.89
print(f"payment_value summed over rows   : R$ {paid_rows:,.2f}")
print(f"payment_value counted once/payment: R$ {paid_once:,.2f}  "
      f"(rows overcount by {paid_rows / paid_once - 1:.1%})")

inst_by_type = pays.groupby("payment_type")["payment_installments"].agg(["min", "max", "count"])
print("\ninstalments per payment type (n = payments):")
print(inst_by_type.to_string())
only_card_over_1 = (pays.loc[pays["payment_type"] != "credit_card", "payment_installments"] == 1).all()
print("every non-credit-card payment has exactly 1 instalment:", only_card_over_1)

# ----------------------------------------------------------------- OLD METHOD REPLAY
head("B. EARLIER METHOD REPLAYED (to confirm it reproduces the earlier printed numbers)")
old = df.groupby("customer_id").agg(
    total_spend=("payment_value", "sum"),
    avg_installments=("payment_installments", "mean"),
    avg_price=("price", "mean"),
    avg_freight=("freight_value", "mean")).reset_index()
old_feats = ["total_spend", "avg_installments", "avg_price", "avg_freight"]
Xo = StandardScaler().fit_transform(old[old_feats])
Xo_s = resample(Xo, n_samples=SAMPLE_N, random_state=SEED)      # replace=True (old default)
old_sil = {}
for k in K_RANGE:
    lab = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(Xo_s)
    old_sil[k] = silhouette_score(Xo_s, lab)
old_k = max(old_sil, key=old_sil.get)
old_lab = KMeans(n_clusters=old_k, random_state=SEED, n_init=10).fit_predict(Xo)
old["cluster"] = old_lab
old_sizes = sorted(np.bincount(old_lab).tolist())
old_means = old.groupby("cluster")["total_spend"].mean().sort_values().round(2).tolist()
print(f"earlier k chosen {old_k} | cluster sizes (sorted) {old_sizes} | earlier version {sorted(OLD_NB['sizes_k2'])} -> "
      f"{flag(old_sizes == sorted(OLD_NB['sizes_k2']))}")
print(f"earlier mean total_spend by cluster (low, high) {old_means} | earlier version "
      f"{sorted(OLD_NB['mean_spend'])} -> {flag(old_means == sorted(OLD_NB['mean_spend']))}")
for k in K_RANGE:
    print(f"   earlier silhouette k={k}: {old_sil[k]:.4f} | earlier version {OLD_NB['sil_sample'][k]:.4f} -> "
          f"{flag(abs(old_sil[k] - OLD_NB['sil_sample'][k]) < 0.0005)}")
r_old = df[["payment_installments", "payment_value"]].corr().iloc[0, 1]
print(f"earlier correlation instalments vs payment_value (n = {len(df):,} rows): {r_old:.4f} | earlier version "
      f"{OLD_NB['corr_inst_value']} -> {flag(abs(r_old - OLD_NB['corr_inst_value']) < 0.00005)}")
print("earlier max total_spend:", f"{old['total_spend'].max():,.2f}")

# ----------------------------------------------------------------- NEW customer table
head("C. NEW CUSTOMER TABLE (customer = customer_unique_id; each payment and item counted once)")
pay_c = pays.groupby("customer_unique_id").agg(
    total_paid=("payment_value", "sum"),
    avg_installments=("payment_installments", "mean"),
    n_payments=("payment_sequential", "size"))
item_c = items.groupby("customer_unique_id").agg(
    avg_price=("price", "mean"),
    avg_freight=("freight_value", "mean"),
    n_items=("order_item_id", "size"))
n_ord_c = df.drop_duplicates("order_id").groupby("customer_unique_id").size().rename("n_orders")
cust = pay_c.join(item_c, how="inner").join(n_ord_c, how="inner").reset_index()
assert len(cust) == n_cust == 92_081
assert cust["customer_unique_id"].is_unique
assert round(cust["total_paid"].sum(), 2) == round(paid_once, 2), "money no longer adds up"
assert cust["n_orders"].sum() == n_orders
feats = ["total_paid", "avg_installments", "avg_price", "avg_freight"]
print(f"customers {len(cust):,} | their total_paid adds up to R$ {cust['total_paid'].sum():,.2f} "
      f"(= payments counted once)")
vc = cust["n_orders"].value_counts().sort_index()
n1, n2 = int(vc.get(1, 0)), int(vc.get(2, 0))
n3 = int(vc[vc.index >= 3].sum())
print(f"orders per customer: 1 order {n1:,} ({n1 / n_cust:.1%}) | 2 orders {n2:,} | 3+ orders {n3:,} "
      f"(max {int(cust['n_orders'].max())})")
print("\nfeature summary (n = customers):")
print(cust[feats].describe(percentiles=[.5, .9, .99]).round(2).to_string())
print("\ntop 5 customers by total_paid (R$):")
print(cust.nlargest(5, "total_paid")[["total_paid", "n_orders", "n_items", "n_payments"]]
      .round(2).to_string(index=False))
print("\ncorrelation between the 4 features (Pearson, n = customers):")
print(cust[feats].corr().round(3).to_string())
r_new = pays[["payment_installments", "payment_value"]].corr().iloc[0, 1]
r_new_sp = pays[["payment_installments", "payment_value"]].corr(method="spearman").iloc[0, 1]
print(f"\ninstalments vs payment_value, n = {len(pays):,} payments: Pearson {r_new:.4f} | Spearman {r_new_sp:.4f}")

# ----------------------------------------------------------------- choose k (rule fixed in advance)
head("D. CHOOSING k (rule: highest silhouette on a 10,000-customer sample, no repeats)")
X = StandardScaler().fit_transform(cust[feats])
idx = np.random.RandomState(SEED).choice(len(X), SAMPLE_N, replace=False)
assert len(np.unique(idx)) == SAMPLE_N
Xs = X[idx]
ks = []
for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(Xs)
    ks.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(Xs, km.labels_)})
ks = pd.DataFrame(ks)
ks.to_csv(OUT / "k_search.csv", index=False)
print(ks.round(4).to_string(index=False))
best_k = int(ks.loc[ks["silhouette"].idxmax(), "k"])
print(f"\nchosen k (highest silhouette) = {best_k}")

# ----------------------------------------------------------------- final model
head("E. FINAL MODEL ON ALL CUSTOMERS")


def fit_labelled(k, seed):
    """KMeans, then renumber clusters so 0 = lowest mean total_paid (labels are arbitrary otherwise)."""
    lab = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X)
    order = cust.groupby(lab)["total_paid"].mean().sort_values().index.tolist()
    return pd.Series(lab).map({c: i for i, c in enumerate(order)}).to_numpy()


t1 = time.time()
cust["cluster"] = fit_labelled(best_k, SEED)
sil_full = sil_final(X, cust["cluster"])
sil_scope = "all" if FULL_SILHOUETTE else f"a {SIL_SAMPLE:,}-customer sample of"
print(f"k = {best_k} | silhouette on {sil_scope} {len(cust):,} customers = {sil_full:.4f} "
      f"(took {time.time() - t1:.0f}s)")
sizes = cust["cluster"].value_counts().sort_index()

prof_mean = cust.groupby("cluster")[feats].mean().round(2)
prof_med = cust.groupby("cluster")[feats].median().round(2)
prof = prof_mean.add_suffix("_mean").join(prof_med.add_suffix("_median"))
prof.insert(0, "customers", sizes)
prof.insert(1, "share_%", (sizes / len(cust) * 100).round(1))
prof.to_csv(OUT / "cluster_profile.csv")
print("\ncluster profile (n = customers; mean and median because the data are skewed):")
print(prof.T.to_string())

mix = (pays.merge(cust[["customer_unique_id", "cluster"]], on="customer_unique_id")
       .groupby(["cluster", "payment_type"]).size().unstack(fill_value=0))
mix_pct = (mix.div(mix.sum(axis=1), axis=0) * 100).round(1)
print("\npayment type mix by cluster (% of payments; n = payments):")
print(pd.concat([mix.sum(axis=1).rename("payments"), mix_pct], axis=1).to_string())

# ----------------------------------------------------------------- stability + k=3 (checks only)
head("F. CHECKS (not used to change the result)")
base = cust["cluster"].to_numpy()
aris = []
for s in STABILITY_SEEDS:
    other = fit_labelled(best_k, s)
    aris.append(adjusted_rand_score(base, other))
    print(f"   seed {s}: sizes {np.bincount(other).tolist()} | agreement with seed {SEED} "
          f"(adjusted Rand, 1.0 = identical) = {aris[-1]:.4f}")
if best_k != 3:
    lab3 = fit_labelled(3, SEED)
    p3 = cust.assign(c3=lab3).groupby("c3")[feats].agg(["mean"]).round(2)
    p3.columns = [a for a, _ in p3.columns]
    p3.insert(0, "customers", np.bincount(lab3))
    print(f"\nfor reference only - k = 3 fitted on all customers (silhouette on a 20,000-customer sample: "
          f"{sil_final(X, lab3):.4f}):")
    print(p3.to_string())
big = cust.nlargest(1, "total_paid")["customer_unique_id"].iloc[0]
Xm = StandardScaler().fit_transform(cust.loc[cust["customer_unique_id"] != big, feats])
lab_m = KMeans(n_clusters=best_k, random_state=SEED, n_init=10).fit_predict(Xm)
print(f"\nwithout the single largest customer (n = {len(Xm):,}): sizes {sorted(np.bincount(lab_m).tolist())} "
      f"vs with {sorted(sizes.tolist())}")

# ----------------------------------------------------------------- old vs new table
head("G. EARLIER vs NOW")
rows = [
    ("Unit called 'customer'", "customer_id (one per ORDER)", "customer_unique_id"),
    ("Units clustered", f"{len(old):,}", f"{len(cust):,}"),
    ("Total paid used (R$)", f"{paid_rows:,.2f} (rows)", f"{paid_once:,.2f} (payments once)"),
    ("Largest single total (R$)", f"{old['total_spend'].max():,.2f}", f"{cust['total_paid'].max():,.2f}"),
    ("Sample to choose k", "10,000 WITH repeats", "10,000 WITHOUT repeats"),
    ("k chosen (max silhouette)", f"{old_k}", f"{best_k}"),
    ("Silhouette on sample at chosen k", f"{old_sil[old_k]:.4f}", f"{ks.loc[ks['silhouette'].idxmax(), 'silhouette']:.4f}"),
    ("Silhouette, final model", f"{OLD_NB['sil_final']:.4f} (all {len(old):,}, earlier version)", f"{sil_full:.4f} ({'all' if FULL_SILHOUETTE else 'sample of ' + format(SIL_SAMPLE, ',')})"),
    ("Cluster sizes (low spend first)", f"{old_sizes[::-1]}", f"{sizes.tolist()}"),
    ("Instalments vs payment_value r", f"{r_old:.4f} (n={len(df):,} rows)", f"{r_new:.4f} (n={len(pays):,} payments)"),
    ("Payment type", "profiled afterwards, not an input", "profiled afterwards, not an input"),
]
ovn = pd.DataFrame(rows, columns=["item", "earlier", "now"])
ovn.to_csv(OUT / "old_vs_new.csv", index=False)
print(ovn.to_string(index=False))

# ----------------------------------------------------------------- charts
head("H. CHARTS")
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
colors = ["#1f4e79", "#e69f00", "#2a9d8f", "#9b5de5", "#c1121f"]

# 1 orders per customer
fig, ax = plt.subplots(figsize=(7, 4.5))
labels = ["1 order", "2 orders", "3+ orders"]
vals = [n1, n2, n3]
ax.bar(labels, vals, color=colors[0])
for i, v in enumerate(vals):
    ax.text(i, v, f"{v:,}\n({v / n_cust:.1%})", ha="center", va="bottom")
ax.set_ylim(0, max(vals) * 1.18)
ax.set_ylabel("Customers")
ax.set_title(f"Orders per customer (n = {n_cust:,} customers, {n_orders:,} orders)")
fig.text(0.01, 0.01, "customer = customer_unique_id", fontsize=8, color="grey")
fig.tight_layout(); fig.savefig(OUT / "01_orders_per_customer.png", dpi=200); plt.close(fig)

# 2 choosing k
fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
axs[0].plot(ks["k"], ks["inertia"], marker="o", color=colors[0])
axs[0].set_xlabel("k"); axs[0].set_ylabel("Inertia (within-cluster squared distance)")
axs[0].set_title("Elbow (inertia by k)")
axs[1].plot(ks["k"], ks["silhouette"], marker="s", color=colors[1])
axs[1].set_xlabel("k"); axs[1].set_ylabel("Silhouette (higher = better separated)")
axs[1].set_title(f"Silhouette (highest at k = {best_k})")
for a in axs:
    a.set_xticks(list(K_RANGE)); a.grid(alpha=.3)
fig.suptitle(f"Choosing k (n = {SAMPLE_N:,} customers sampled without repeats, 4 scaled features)")
fig.tight_layout(); fig.savefig(OUT / "02_choosing_k.png", dpi=200); plt.close(fig)

# 3 scatter
fig, ax = plt.subplots(figsize=(8, 5.5))
for c in range(best_k):
    d = cust[cust["cluster"] == c]
    ax.scatter(d["total_paid"].clip(lower=1), d["avg_installments"], s=3, alpha=.25,
               color=colors[c % 5], label=f"cluster {c} (n = {len(d):,})", rasterized=True)
ax.set_xscale("log")
ax.set_xlabel("Total paid per customer (R$, log scale)")
ax.set_ylabel("Average instalments per payment")
ax.set_title(f"Customers by total paid and instalments (n = {len(cust):,} customers)")
ax.legend(markerscale=4); ax.grid(alpha=.3)
fig.text(0.01, 0.01, "Clusters also use avg item price and avg freight, which are not drawn here.",
         fontsize=8, color="grey")
fig.tight_layout(); fig.savefig(OUT / "03_segments_scatter.png", dpi=200); plt.close(fig)

# 4 profile (medians)
fig, axs = plt.subplots(1, 4, figsize=(12, 4))
titles = ["Total paid (R$)", "Instalments per payment", "Item price (R$)", "Freight per item (R$)"]
for a, f, t in zip(axs, feats, titles):
    v = prof_med[f]
    a.bar([str(i) for i in v.index], v.values, color=[colors[i % 5] for i in v.index])
    for i, x in enumerate(v.values):
        a.text(i, x, f"{x:,.1f}", ha="center", va="bottom", fontsize=9)
    a.set_title(t); a.set_xlabel("Cluster"); a.set_ylim(0, v.max() * 1.15)
fig.suptitle("Cluster medians of per-customer values (customers: " + " / ".join(f"cluster {i} n = {n:,}" for i, n in sizes.items()) + ")")
fig.tight_layout(); fig.savefig(OUT / "04_cluster_profiles_median.png", dpi=200); plt.close(fig)

# 5 payment mix
fig, ax = plt.subplots(figsize=(8, 4.5))
pay_colors = ["#6c757d", "#2a9d8f", "#c1121f", "#9b5de5"]
bottom = np.zeros(len(mix_pct))
for j, t in enumerate(mix_pct.columns):
    ax.barh([f"cluster {i} (n = {int(mix.loc[i].sum()):,} payments)" for i in mix_pct.index],
            mix_pct[t].values, left=bottom, color=pay_colors[j], label=t)
    bottom += mix_pct[t].values
ax.set_xlabel("% of payments"); ax.set_xlim(0, 100)
ax.set_title(f"Payment type mix by cluster (n = {len(pays):,} payments)")
fig.text(0.01, 0.01, "Payment type was not a clustering input; it is profiled after clustering.", fontsize=8, color="grey")
ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(.5, -.35), frameon=False)
fig.tight_layout(); fig.savefig(OUT / "05_payment_mix_by_cluster.png", dpi=200); plt.close(fig)
print("saved 5 PNGs and 3 CSVs in", OUT.relative_to(BASE))



# -----------------------------------------------------------------
# README draft (numbers filled from this run)
lines = [
    "## Module 5 - Customer payment-behaviour segments (K-Means)",
    "",
    f"- Data: the file has {len(df):,} rows. They hold {len(items):,} items and "
    f"{len(pays):,} payments from {n_orders:,} distinct orders placed by "
    f"{n_cust:,} customers (customer_unique_id). Each payment and item is counted once per customer.",
    f"- {n1 / n_cust:.1%} of customers placed one order, so this is mostly a one-order view of each customer.",
    "- Features (scaled): total paid, average instalments per payment, average item price, average freight.",
    "  Payment type is NOT an input; it is only profiled after clustering.",
    f"- k chosen as the highest silhouette on a {SAMPLE_N:,}-customer sample (k = {best_k}); "
    f"silhouette of the final model {sil_full:.2f} "
    f"({'all customers' if FULL_SILHOUETTE else 'on a ' + format(SIL_SAMPLE, ',') + '-customer sample'}). "
    f"At k = {best_k} the silhouette on the {SAMPLE_N:,}-customer sample was "
    f"{ks['silhouette'].max():.2f}; the final model's value above is on a different sample, "
    "so the two are not directly comparable. "
    "Inertia falls steadily as k rises, so the elbow plot does not point to one k.",
    f"- Clusters (n = customers): " + "; ".join(
        f"cluster {i}: {n:,} ({n / len(cust):.1%}), "
        f"median total paid R$ {prof_med.loc[i, 'total_paid']:,.2f}, "
        f"median instalments {prof_med.loc[i, 'avg_installments']:.1f}"
        for i, n in sizes.items()
    ) + ".",
    f"- Re-running with {len(STABILITY_SEEDS)} other seeds gave agreement (adjusted Rand) of "
    f"{min(aris):.2f} to {max(aris):.2f} with the main run.",
    "- Limits: total paid is strongly related to average item price "
    "(Pearson r = 0.90, n = 92,081 customers), instalments above 1 occur only on credit-card "
    "payments, spending is heavily skewed, and silhouette measures separation only "
    "(it says nothing about usefulness). One dataset, 2016-2018.",
    "- Possible use (not tested): grouping customers by spend and instalment habits for descriptive reporting.",
]
(OUT / "README_module5_draft.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("wrote README_module5_draft.md")
print("\nDONE")