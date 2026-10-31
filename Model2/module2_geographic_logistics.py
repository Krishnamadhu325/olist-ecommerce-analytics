"""
Module 2 (corrected): Geographic logistics aggregation, Olist.

What changed vs the earlier version
- Every table is one row per ORDER (not per item/payment row).
- Late = delivered timestamp is after the estimated date (same rule as Model 1: 8.10%).
- Regions: North, Northeast, Southeast, South, Center-West (no "Other").
- Missing delivery dates are counted and left out of the rates, never silently "not late".
- Small-n states are flagged and shown with n and a 95% range.

Folder layout expected (relative to this file, inside Model2/):
    data/olist_order_level.csv               (one row per order; copy it from Model1/data/ after running 01_build_order_table.py)
    data/Brazilian-Ecommerce-Cleaned.xlsx    (item x payment rows, needed ONLY for seller state)
    outputs/module2/                         (created automatically)

Run from the repository root:
    python Model2\module2_geographic_logistics.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save charts to files, no pop-up windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── SETTINGS ──────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ORDER_CSV = BASE / "data" / "olist_order_level.csv"
ITEM_XLSX = BASE / "data" / "Brazilian-Ecommerce-Cleaned.xlsx"
OUT = BASE / "outputs" / "module2"
OUT.mkdir(parents=True, exist_ok=True)

SMALL_N = 200  # states with fewer orders than this are flagged (a judgment call, change if you like)

REGION_OF = {
    "AC": "North", "AP": "North", "AM": "North", "PA": "North", "RO": "North", "RR": "North", "TO": "North",
    "AL": "Northeast", "BA": "Northeast", "CE": "Northeast", "MA": "Northeast", "PB": "Northeast",
    "PE": "Northeast", "PI": "Northeast", "RN": "Northeast", "SE": "Northeast",
    "ES": "Southeast", "MG": "Southeast", "RJ": "Southeast", "SP": "Southeast",
    "PR": "South", "RS": "South", "SC": "South",
    "DF": "Center-West", "GO": "Center-West", "MT": "Center-West", "MS": "Center-West",
}
REGION_ORDER = ["North", "Northeast", "Southeast", "South", "Center-West"]

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 100)


def head(text):
    print("\n" + "=" * 100 + f"\n{text}\n" + "=" * 100)


def wilson(k, n, z=1.96):
    """95% range for a rate k/n. Wider when n is small."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return np.maximum(centre - half, 0), np.minimum(centre + half, 1)


def summarise(df, by):
    """One row per group. df MUST already be one row per order."""
    t = df.groupby(by).agg(
        n_orders=("order_id", "size"),
        late_orders=("is_late", "sum"),
        late_rate=("is_late", "mean"),
        mean_freight_per_order=("freight_total", "mean"),
        mean_delivery_days=("delivery_days", "mean"),
        mean_days_vs_estimate=("days_vs_estimate", "mean"),
        mean_estimate_window_days=("est_window_days", "mean"),
    )
    lo, hi = wilson(t["late_orders"], t["n_orders"])
    t["late_rate_ci95_low"], t["late_rate_ci95_high"] = lo, hi
    t["small_n"] = t["n_orders"] < SMALL_N
    return t


def show(t, name, extra=()):
    """Print a table with percentages and a readable 95% range."""
    p = t.copy()
    p["late_rate_%"] = (p["late_rate"] * 100).round(2)
    p["95%_range"] = (p["late_rate_ci95_low"] * 100).round(1).astype(str) + "-" + (p["late_rate_ci95_high"] * 100).round(1).astype(str)
    cols = ["n_orders", "late_orders", "late_rate_%", "95%_range", "mean_freight_per_order",
            "mean_delivery_days", "mean_estimate_window_days", "mean_days_vs_estimate", "small_n", *extra]
    print(f"\n{name}")
    print(p[cols].round(2).to_string())


# ══ STEP 1. LOAD THE ORDER-LEVEL TABLE (one row per order) ════════════════════
head("STEP 1. Load order-level table and check the grain")
orders = pd.read_csv(ORDER_CSV)
date_cols = ["order_purchase_timestamp", "order_approved_at",
             "order_estimated_delivery_date", "order_delivered_customer_date"]
for c in date_cols:
    orders[c] = pd.to_datetime(orders[c])

# THE ONE-ROW-PER-ORDER ASSERT
assert orders["order_id"].is_unique, "order_id is not unique: this is not one row per order"
print(f"OK: {len(orders):,} rows and {orders['order_id'].nunique():,} distinct orders (must be equal)")

# Missing delivery date -> cannot say late or not late. Count them and leave them out of the rates.
missing = orders["order_delivered_customer_date"].isna()
print(f"Orders with a missing delivery date: {int(missing.sum()):,} (left out of every rate below)")
orders = orders.loc[~missing].copy()

# Late by exact timestamp, rebuilt here and checked against the file's own is_late column.
file_is_late = orders["is_late"].copy()
orders["is_late"] = (orders["order_delivered_customer_date"] > orders["order_estimated_delivery_date"]).astype(int)
assert (orders["is_late"] == file_is_late).all(), "rebuilt late flag differs from is_late in the CSV"
orders["days_vs_estimate"] = (orders["order_delivered_customer_date"] - orders["order_estimated_delivery_date"]).dt.total_seconds() / 86400
orders["delivery_days"] = (orders["order_delivered_customer_date"] - orders["order_purchase_timestamp"]).dt.total_seconds() / 86400
print(f"Late orders: {orders['is_late'].sum():,} of {len(orders):,} = {orders['is_late'].mean():.2%}")
print(f"Orders with status canceled still in the table: {(orders['order_status'] == 'canceled').sum()} kept: they carry a delivery date")

orders["region"] = orders["customer_state"].map(REGION_OF)
unmapped = orders.loc[orders["region"].isna(), "customer_state"].unique()
assert len(unmapped) == 0, f"States missing from REGION_OF: {unmapped}"

# ══ STEP 2. CUSTOMER-STATE TABLE (one row per order) ══════════════════════════
head("STEP 2. Customer-state table (unit = order)")
cust = summarise(orders, "customer_state")
assert cust["n_orders"].sum() == len(orders)
cust = cust.sort_values("late_rate", ascending=False)
show(cust, f"Customer states, sorted by late rate (small_n = True means fewer than {SMALL_N} orders, do not rank on it)")
cust.to_csv(OUT / "customer_state_table.csv")

# ══ STEP 3. REGION TABLE (five regions) ═══════════════════════════════════════
head("STEP 3. Region table (customer region, unit = order)")
reg = summarise(orders, "region").reindex(REGION_ORDER)
assert reg["n_orders"].sum() == len(orders)
show(reg, "Customer regions")
reg.to_csv(OUT / "region_table.csv")

# Same-state vs cross-state shipping (this is a flag, NOT a distance measure)
cross = summarise(orders, "any_cross_state")
cross.index = cross.index.map({0: "all items from customer's own state", 1: "at least one item from another state"})
show(cross, "Cross-state shipping flag (order-level)")
cross.to_csv(OUT / "cross_state_table.csv")

# ══ STEP 4. SELLER-STATE TABLE ════════════════════════════════════════════════
head("STEP 4. Seller-state table (unit = order, one seller state per order)")
need = ["order_id", "order_item_id", "seller_id", "seller_state", "customer_state", "freight_value",
        "order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"]
rows = pd.read_excel(ITEM_XLSX, usecols=need)  # slow (about 1 minute), it is a 113k-row Excel file
print(f"Item x payment rows loaded: {len(rows):,}")

# Same orders in both files? If not, stop.
assert set(rows["order_id"]) == set(pd.read_csv(ORDER_CSV, usecols=["order_id"])["order_id"]), \
    "order_id sets differ between the xlsx and the order-level CSV"

# Remove the payment duplication: one row per item
items = rows.drop_duplicates(["order_id", "order_item_id"])
print(f"Items: {len(items):,}  (rows / items = payments repeat each item)")

per_order = items.groupby("order_id").agg(n_seller_states=("seller_state", "nunique"),
                                          seller_state=("seller_state", "first"))
sell = orders.merge(per_order, on="order_id", how="left", validate="one_to_one")
assert sell["seller_state"].notna().all(), "some orders have no seller state"

multi = sell["n_seller_states"] > 1
print(f"Orders shipped from 2+ seller STATES: {int(multi.sum()):,} -> left out of the seller-state table "
      "(we cannot say which state caused the delay).")
print(f"   their late rate {sell.loc[multi, 'is_late'].mean():.2%} vs {sell.loc[~multi, 'is_late'].mean():.2%} for the rest")
sell = sell.loc[~multi].copy()
assert sell["order_id"].is_unique  # still one row per order
print(f"Orders in seller-state table: {len(sell):,}")

seller = summarise(sell, "seller_state").sort_values("n_orders", ascending=False)
assert seller["n_orders"].sum() == len(sell)
seller["share_of_orders_%"] = (seller["n_orders"] / len(sell) * 100).round(1)
# How many different sellers sit behind each state? Few sellers = the state's rate may be 1-2 sellers, not the state.
seller["n_distinct_sellers"] = items[items["order_id"].isin(sell["order_id"])].groupby("seller_state")["seller_id"].nunique()
show(seller, f"Seller states, sorted by order count (small_n = fewer than {SMALL_N} orders)", extra=["share_of_orders_%", "n_distinct_sellers"])
print("\nTop 3 seller states hold", f"{seller['n_orders'].head(3).sum() / len(sell):.1%}", "of these orders")
seller.to_csv(OUT / "seller_state_table.csv")

# ══ STEP 5. OLD METHOD REPLAY (rows counted as orders) ════════════════════════
head("STEP 5. Old method replayed from the same file, for the old-vs-new comparison")
old = rows.copy()
old["delay_days"] = (old["order_delivered_customer_date"] - old["order_estimated_delivery_date"]).dt.days
old["delivery_days"] = (old["order_delivered_customer_date"] - old["order_purchase_timestamp"]).dt.days
print(f"Rows with NaN delay_days (old code counted these as 'not late'): {int(old['delay_days'].isna().sum()):,}")
north = ["AM", "PA", "RO", "RR", "AP", "AC", "TO"]
southeast = ["SP", "RJ", "MG", "ES"]
old["region_old"] = old["customer_state"].apply(lambda s: "North" if s in north else ("Southeast" if s in southeast else "Other"))


def old_summary(g):
    return pd.Series({
        "old_rows": g["delay_days"].count(),
        "old_late_rate": (g["delay_days"] > 0).mean(),
        "old_freight": g["freight_value"].mean(),
        "old_delivery_days": g["delivery_days"].mean(),
        "old_avg_delay_days": g["delay_days"].mean(),
    })


old_cust = old.groupby("customer_state").apply(old_summary, include_groups=False)
old_reg = old.groupby("region_old").apply(old_summary, include_groups=False)
old_sell = old.groupby("seller_state").apply(old_summary, include_groups=False)
old_all = old_summary(old)

# Check the replay against numbers printed in the old notebook
checks = [
    ("old rows loaded", len(old), 113390, 0),
    ("old North late rate", old_reg.loc["North", "old_late_rate"], 0.086324, 1e-5),
    ("old Southeast late rate", old_reg.loc["Southeast", "old_late_rate"], 0.058741, 1e-5),
    ("old AL late rate", old_cust.loc["AL", "old_late_rate"], 0.212815, 1e-5),
    ("old SP seller rows", old_sell.loc["SP", "old_rows"], 80870, 0),
    ("old AM seller avg delivery days", old_sell.loc["AM", "old_delivery_days"], 47.333333, 1e-4),
]
for name, got, want, tol in checks:
    print(f"  {'OK      ' if abs(got - want) <= tol else 'MISMATCH'}  {name}: replay {got:.6g} vs the old notebook {want:.6g}")

# ══ STEP 6. OLD VS NEW COMPARISON ═════════════════════════════════════════════
head("STEP 6. Old vs new (old counted rows, new counts orders)")


def pair(old_row, new_row, label):
    return {
        "group": label,
        "old_rows": old_row["old_rows"], "new_orders": new_row["n_orders"],
        "old_late_%": old_row["old_late_rate"] * 100, "new_late_%": new_row["late_rate"] * 100,
        "old_freight": old_row["old_freight"], "new_freight": new_row["mean_freight_per_order"],
        "old_deliv_days": old_row["old_delivery_days"], "new_deliv_days": new_row["mean_delivery_days"],
        "old_avg_delay": old_row["old_avg_delay_days"], "new_vs_estimate": new_row["mean_days_vs_estimate"],
    }


new_all = summarise(orders.assign(_all="ALL"), "_all").iloc[0]
comp = [pair(old_all, new_all, "ALL")]
for r in ["North", "Southeast"]:
    comp.append(pair(old_reg.loc[r], reg.loc[r], f"{r} (region)"))
other = orders[orders["region"].isin(["Northeast", "South", "Center-West"])].assign(_g="x")
comp.append(pair(old_reg.loc["Other"], summarise(other, "_g").iloc[0], "'Other' = NE+S+CW"))
for r in ["Northeast", "South", "Center-West"]:
    comp.append({"group": f"{r} (region, new only)", "new_orders": reg.loc[r, "n_orders"], "new_late_%": reg.loc[r, "late_rate"] * 100,
                 "new_freight": reg.loc[r, "mean_freight_per_order"], "new_deliv_days": reg.loc[r, "mean_delivery_days"],
                 "new_vs_estimate": reg.loc[r, "mean_days_vs_estimate"]})
for s in cust.index:
    comp.append(pair(old_cust.loc[s], cust.loc[s], f"cust {s}"))
for s in seller.index:
    comp.append(pair(old_sell.loc[s], seller.loc[s], f"seller {s}"))
comp = pd.DataFrame(comp)
print(comp.round(2).to_string(index=False))
comp.to_csv(OUT / "old_vs_new.csv", index=False)

# ══ STEP 7. CHARTS ════════════════════════════════════════════════════════════
head("STEP 7. Charts")


def bar_late(t, title, fname):
    d = t.sort_values("late_rate")
    colors = ["lightgrey" if sn else "tab:blue" for sn in d["small_n"]]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.28 * len(d))))
    xerr = np.vstack([d["late_rate"] - d["late_rate_ci95_low"], d["late_rate_ci95_high"] - d["late_rate"]]) * 100
    ax.barh(d.index.astype(str) + "  (n=" + d["n_orders"].astype(int).astype(str) + ")", d["late_rate"] * 100,
            color=colors, xerr=xerr, error_kw={"elinewidth": 0.8, "capsize": 2})
    ax.axvline(orders["is_late"].mean() * 100, color="red", linestyle="--", linewidth=1, label=f"overall {orders['is_late'].mean():.2%}")
    ax.set_xlabel(f"Late rate, % of orders (grey = fewer than {SMALL_N} orders; bars = 95% range)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=200)
    plt.close(fig)


bar_late(cust, "Late rate by customer state (unit = order)", "customer_state_late_rate.png")
bar_late(reg, "Late rate by customer region (unit = order)", "region_late_rate.png")
bar_late(seller, "Late rate by seller state (orders from one seller state)", "seller_state_late_rate.png")
print("Saved charts and CSV tables to:", OUT)
print("No map was built in this module.")
