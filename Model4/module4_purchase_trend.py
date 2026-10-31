"""
Module 4 - Monthly purchase trend (rebuilt at ORDER grain).

Run from the repository root:
    python Model4\module4_purchase_trend.py

Reads : Model4/data/Brazilian-Ecommerce-Cleaned.xlsx   (nothing else)
Writes: Model4/outputs/module4/  (4 PNG charts, CSV tables, results_summary.txt)

Unit rules used everywhere in this script:
  row   = one line of the xlsx (item x payment)
  item  = one distinct order_unique_id
  order = one distinct order_id
  month = month of order_purchase_timestamp
"""
from pathlib import Path
import calendar

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # save PNGs only, no pop-up windows
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from statsmodels.tsa.seasonal import seasonal_decompose

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "Brazilian-Ecommerce-Cleaned.xlsx"
OUT = BASE / "outputs" / "module4"
MIN_ORDERS = 100        # judgment call, not a statistical rule: a month needs >= 100 orders.
COLS = ["order_id", "order_unique_id", "order_purchase_timestamp", "price", "order_status"]

LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def head(t):
    say("\n" + "=" * 8, t, "=" * 8)


def ok(flag):
    return "OK" if flag else "DIFFERENT"


assert DATA.exists(), f"Missing {DATA}. Put the xlsx in data/ or edit DATA at the top."
OUT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ 1. LOAD + GRAIN
head("1. LOAD AND GRAIN CHECKS")
df = pd.read_excel(DATA, usecols=COLS)
df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
n_rows, n_items, n_orders = len(df), df["order_unique_id"].nunique(), df["order_id"].nunique()
say(f"rows {n_rows:,} | items {n_items:,} | orders {n_orders:,}")
assert (n_rows, n_items, n_orders) == (113390, 108640, 95128), "grain counts changed - stop"

g = df.groupby("order_id")
assert (g["order_purchase_timestamp"].nunique() == 1).all(), "purchase time changes inside an order"
assert (g["order_status"].nunique() == 1).all(), "order_status changes inside an order"

orders = df.drop_duplicates("order_id")
items = df.drop_duplicates("order_unique_id")
assert orders["order_id"].is_unique and len(orders) == 95128
assert items["order_unique_id"].is_unique and len(items) == 108640
price_total = round(items["price"].sum(), 2)
say(f"item price counted once per item: R$ {price_total:,.2f}")
assert price_total == 13046300.07, "item price total differs from the verified 13,046,300.07"
say("grain asserts passed (rows = items x payments is already verified from Module 3)")

# ------------------------------------------------------------------ 2. MONTHLY TABLE
head("2. MONTHLY TABLE (orders = distinct order_id by purchase month)")
key = lambda d: d["order_purchase_timestamp"].dt.to_period("M")
monthly = pd.DataFrame({
    "rows": df.groupby(key(df)).size(),
    "items": items.groupby(key(items)).size(),
    "orders": orders.groupby(key(orders)).size(),
    "item_price_R$": items.groupby(key(items))["price"].sum().round(2),
})
full = pd.period_range(monthly.index.min(), monthly.index.max(), freq="M")
monthly = monthly.reindex(full)
missing = [str(m) for m in monthly.index[monthly["orders"].isna()]]
monthly = monthly.fillna(0)
for c in ["rows", "items", "orders"]:
    monthly[c] = monthly[c].astype(int)
monthly["rows_per_order"] = (monthly["rows"] / monthly["orders"].replace(0, np.nan)).round(3)
monthly["orders_per_calendar_day"] = [
    round(o / calendar.monthrange(p.year, p.month)[1], 1) for p, o in zip(monthly.index, monthly["orders"])]
monthly["item_price_per_order_R$"] = (monthly["item_price_R$"] / monthly["orders"].replace(0, np.nan)).round(2)

# window rule: the unbroken run of months, ending at the last month, that each have >= MIN_ORDERS
good = (monthly["orders"] >= MIN_ORDERS).to_numpy()
assert good[-1], "last month has fewer than the minimum orders"
start = len(good)
while start > 0 and good[start - 1]:
    start -= 1
monthly["in_window"] = False
monthly.iloc[start:, monthly.columns.get_loc("in_window")] = True
monthly["orders_change_vs_prev_month_%"] = (monthly["orders"].pct_change() * 100).round(1)
monthly.loc[~monthly["in_window"], "orders_change_vs_prev_month_%"] = np.nan
monthly.loc[monthly.index[start], "orders_change_vs_prev_month_%"] = np.nan

W = monthly[monthly["in_window"]]
say(monthly.drop(columns=["orders_per_calendar_day"]).to_string())
say(f"\nmissing calendar months (no orders at all): {missing}")
say(f"rule: unbroken run of months with >= {MIN_ORDERS} orders, ending at the last month")
say(f"window = {W.index[0]} to {W.index[-1]}  ({len(W)} months)")
n_w, items_w = int(W["orders"].sum()), int(W["items"].sum())
say(f"orders in window {n_w:,} of {n_orders:,} | left out: {n_orders - n_w:,} orders "
    f"({', '.join(str(m) for m in monthly.index[~monthly['in_window']])})")
say(f"items in window {items_w:,} | item price in window R$ {W['item_price_R$'].sum():,.2f}")
assert str(W.index[0]) == "2017-01" and len(W) == 20 and n_w == 94859 and items_w == 108322, \
    "window differs from the one seen in the first run - stop and check"
assert int(monthly["rows"].sum()) == n_rows and int(monthly["orders"].sum()) == n_orders
say(f"rows per order, window: min {W['rows_per_order'].min()} / max {W['rows_per_order'].max()} "
    f"(counting rows overstates orders by 16-29% and the gap is not constant)")
nd = orders[orders["order_status"] != "delivered"]
say(f"orders whose status is not 'delivered' (kept in the file): {len(nd)} "
    f"-> in window: {int((nd['order_purchase_timestamp'].dt.to_period('M') >= W.index[0]).sum())}")
say(nd["order_status"].value_counts().to_string())
say(f"first purchase {df['order_purchase_timestamp'].min()} | last purchase {df['order_purchase_timestamp'].max()}")
monthly.to_csv(OUT / "monthly_table.csv", index_label="purchase_month")

# ------------------------------------------------------------------ 3. SHAPE OF THE SERIES (descriptive)
head("3. SHAPE OF THE ORDER SERIES (descriptive only)")
peak_m = W["orders"].idxmax()
say(f"highest month: {peak_m} with {int(W['orders'].max()):,} orders (unique: "
    f"{int((W['orders'] == W['orders'].max()).sum()) == 1})")
y17 = W[W.index.year == 2017]
y18 = W[W.index.year == 2018]
say(f"2017 months in window: {y17.index[0]} {int(y17['orders'].iloc[0]):,} -> {y17.index[-1]} {int(y17['orders'].iloc[-1]):,}")
say(f"2018 Jan-Aug: min {int(y18['orders'].min()):,} ({y18['orders'].idxmin()}), "
    f"max {int(y18['orders'].max()):,} ({y18['orders'].idxmax()}), mean {y18['orders'].mean():,.0f}")
say(f"Jan 2018 / Nov 2017 = {y18['orders'].iloc[0] / W.loc['2017-11', 'orders']:.3f}")
mx_p = W["item_price_R$"].idxmax()
say(f"highest month by item price: {mx_p} (R$ {W['item_price_R$'].max():,.0f}); "
    f"Nov 2017 item price R$ {W.loc['2017-11', 'item_price_R$']:,.0f}")

yoy = []
for m in range(1, 9):
    a, b = int(W.loc[f"2017-{m:02d}", "orders"]), int(W.loc[f"2018-{m:02d}", "orders"])
    yoy.append({"month": calendar.month_abbr[m], "orders_2017": a, "orders_2018": b,
                "difference": b - a, "ratio_2018_over_2017": round(b / a, 2)})
yoy = pd.DataFrame(yoy)
say("\nsame calendar month, 2017 vs 2018 (only Jan-Aug exist in both years):")
say(yoy.to_string(index=False))
say("note: early-2017 months start from a very small base, so large ratios there are a base effect")
yoy.to_csv(OUT / "yoy_jan_aug.csv", index=False)

# ------------------------------------------------------------------ 4. NOV 2017 DAILY + END-OF-DATA CHECK
head("4. DAILY CHECKS (descriptive; no cause is tested)")
d = orders.assign(day=orders["order_purchase_timestamp"].dt.normalize())
nov = d[d["day"].dt.to_period("M") == "2017-11"].groupby("day").size()
nov_total = int(nov.sum())
top_day = nov.idxmax()
say(f"Nov 2017: {nov_total:,} orders on {len(nov)} days | median day {int(nov.median())} | "
    f"top day {top_day.date()} = {int(nov.max()):,} orders ({nov.max() / nov_total:.1%} of the month)")
say("top 5 days:\n" + nov.sort_values(ascending=False).head(5).to_string())
ex = (nov_total - int(nov.max())) / (len(nov) - 1)
say(f"Nov 2017 orders per day WITHOUT the top day: {ex:.1f}")
for lab in ["2017-10", "2017-12", "2018-01"]:
    say(f"   {lab} orders per calendar day: {monthly.loc[lab, 'orders_per_calendar_day']}")
say("Black Friday 2017 fell on 24 Nov 2017 (calendar fact). The data has no promotion flag, "
    "so it can show WHEN the spike happened, not WHY.")
say(f"purchase dates in the file are on or before {d['day'].max().date()}")
last = d[d["day"].dt.to_period("M") == "2018-08"].groupby("day").size()
say(f"Aug 2018 daily orders, last 10 days present:\n{last.tail(10).to_string()}")
say(f"Aug 2018 median day: {int(last.median())}")
nov_daily = nov.rename("orders").reset_index().rename(columns={"day": "date"})
nov_daily.to_csv(OUT / "nov2017_daily.csv", index=False)

# ------------------------------------------------------------------ 5. OLD METHOD REPLAY
head("5. EARLIER METHOD REPLAY (rows, period=6) - to show why its claims are not supported")
old = df.groupby(key(df)).size()
old.index = old.index.to_timestamp()
old = old.astype(float)
dec = seasonal_decompose(old, model="additive", period=6)
sm = dec.seasonal.copy()
sm.index = sm.index.month
mean_by_month = sm.groupby(sm.index).mean()
mx, mn = mean_by_month.max(), mean_by_month.min()
peak_ties = [int(m) for m in mean_by_month.index[np.isclose(mean_by_month, mx)]]
trough_ties = [int(m) for m in mean_by_month.index[np.isclose(mean_by_month, mn)]]
tr = dec.trend.dropna()
say(f"earlier series: {len(old)} points, sum {int(old.sum()):,} (these are ROWS)")
say(f"{'rows total 113,390':<46}{ok(int(old.sum()) == 113390)}")
say(f"{'Nov 2017 rows 8,683':<46}{ok(int(old['2017-11-01']) == 8683)}")
say(f"{'trend first 1789.17 / last 7702.33':<46}{ok(round(tr.iloc[0], 2) == 1789.17 and round(tr.iloc[-1], 2) == 7702.33)}")
say(f"{'earlier printed peak May / trough June':<46}{ok(int(mean_by_month.idxmax()) == 5 and int(mean_by_month.idxmin()) == 6)}")
say(f"months tied for highest seasonal mean ({mx:.2f}): {[calendar.month_name[m] for m in peak_ties]}")
say(f"months tied for lowest seasonal mean ({mn:.2f}): {[calendar.month_name[m] for m in trough_ties]}")
say("-> May and June were printed only because idxmax/idxmin return the first of equal values.")
pos0 = [old.index[i].strftime("%b %Y") for i in range(0, len(old), 6)]
say(f"positions 0, 6, 12, 18 of the series (all get the SAME seasonal value): {pos0}")
say("-> the 'seasonal' panel repeats every 6 ROWS by construction; with Nov 2016 missing these rows are not 6 calendar months apart.")
try:
    seasonal_decompose(W["orders"].astype(float), model="additive", period=12)
    say("period=12 on the window: ran (unexpected)")
except ValueError as e:
    say(f"period=12 on the {len(W)}-month window -> ValueError: {e}")
say("-> a yearly pattern cannot be estimated from fewer than 2 full years; Sep-Dec exist only once.")

# ------------------------------------------------------------------ 6. OLD vs NEW
head("6. EARLIER vs NOW")
old_vs_new = pd.DataFrame([
    ["what is counted", "rows (item x payment)", "distinct orders (order_id)"],
    ["total counted", f"{int(old.sum()):,} rows, 22 months", f"{n_w:,} orders, {len(W)} months (all {n_orders:,} orders in table)"],
    ["months used", "Oct 2016-Aug 2018 with Nov 2016 missing", f"{W.index[0]} to {W.index[-1]}, unbroken, each >= {MIN_ORDERS} orders"],
    ["Nov 2017", f"{int(old['2017-11-01']):,} rows", f"{int(W.loc['2017-11', 'orders']):,} orders"],
    ["method", "seasonal_decompose, additive, period=6 (chosen, not detected)", "no decomposition: monthly table, same-month 2017 vs 2018, daily check"],
    ["peak / trough month", "May / June (tie-break of equal values)", "no seasonal claim made"],
    ["trend statement", "'upward' (last trend value > first)", f"rise through 2017, then {int(y18['orders'].min()):,}-{int(y18['orders'].max()):,} orders a month in Jan-Aug 2018"],
    ["revenue", "not measured", "item price per month, shown separately (freight excluded)"],
], columns=["item", "old", "new"])
say(old_vs_new.to_string(index=False))
old_vs_new.to_csv(OUT / "old_vs_new.csv", index=False)

# ------------------------------------------------------------------ 7. CHARTS
head("7. CHARTS")
src = "Source: Brazilian-Ecommerce-Cleaned.xlsx; month = order_purchase_timestamp"
mfmt = mdates.DateFormatter("%b %Y")
thou = FuncFormatter(lambda v, _: f"{v:,.0f}")
xs = W.index.to_timestamp()
AUG_NOTE = "Aug 2018 may be incomplete (daily orders fall from 22 Aug)."

# 01 monthly orders
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.plot(xs, W["orders"], marker="o", linewidth=2, color="#1f4e79")
ax.annotate(f"{peak_m}: {int(W['orders'].max()):,}", (peak_m.to_timestamp(), W["orders"].max()),
            textcoords="offset points", xytext=(-10, 10), ha="right", fontsize=9,
            arrowprops=dict(arrowstyle="-", color="grey"))
ax.set_title(f"Orders per month, {W.index[0].strftime('%b %Y')} to {W.index[-1].strftime('%b %Y')} "
             f"(n = {n_w:,} orders, {len(W)} months)", fontsize=13, fontweight="bold")
ax.set_ylabel("Orders (distinct order_id)")
ax.set_xlabel("Purchase month")
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2)); ax.xaxis.set_major_formatter(mfmt)
ax.yaxis.set_major_formatter(thou); ax.grid(True, linestyle="--", alpha=0.4); ax.set_ylim(0, W["orders"].max() * 1.12)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
fig.text(0.01, 0.01, src + ". Oct-Dec 2016 left out (Nov 2016 has 0 orders, which breaks the run).\n" + AUG_NOTE, fontsize=8)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(OUT / "01_orders_per_month.png", dpi=200); plt.close(fig)

# 02 same-month comparison
n_yoy = int(yoy["orders_2017"].sum() + yoy["orders_2018"].sum())
fig, ax = plt.subplots(figsize=(11, 5.5))
x = np.arange(len(yoy)); w = 0.38
ax.bar(x - w / 2, yoy["orders_2017"], w, label="2017", color="#9ecae1")
ax.bar(x + w / 2, yoy["orders_2018"], w, label="2018", color="#1f4e79")
for i, r in yoy.iterrows():
    ax.text(i + w / 2, r["orders_2018"], f"x{r['ratio_2018_over_2017']:.2f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(yoy["month"])
ax.set_title(f"Orders in the same calendar month, 2017 vs 2018, Jan to Aug (n = {n_yoy:,} orders)",
             fontsize=13, fontweight="bold")
ax.set_ylabel("Orders (distinct order_id)"); ax.set_xlabel("Purchase month")
ax.yaxis.set_major_formatter(thou); ax.legend(loc="upper center", ncol=2); ax.set_ylim(0, yoy["orders_2018"].max() * 1.18); ax.grid(True, axis="y", linestyle="--", alpha=0.4)
fig.text(0.01, 0.01, src + ". Labels = 2018 orders divided by 2017 orders.\nEarly-2017 months start from a small base, so large ratios there are a base effect. " + AUG_NOTE, fontsize=8)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(OUT / "02_same_month_2017_vs_2018.png", dpi=200); plt.close(fig)

# 03 Nov 2017 daily
fig, ax = plt.subplots(figsize=(11, 5.5))
cols = ["#d95f02" if dd == top_day else "#1f4e79" for dd in nov.index]
ax.bar(nov.index, nov.values, color=cols)
ax.annotate(f"{top_day.strftime('%d %b')}: {int(nov.max()):,}", (top_day, nov.max()),
            textcoords="offset points", xytext=(16, -4), fontsize=9)
ax.set_title(f"Orders per day, Nov 2017 (n = {nov_total:,} orders)", fontsize=13, fontweight="bold")
ax.set_ylabel("Orders (distinct order_id)"); ax.set_xlabel("Day of November 2017 (purchase date)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d")); ax.yaxis.set_major_formatter(thou)
ax.set_xlim(nov.index[0] - pd.Timedelta(hours=12), nov.index[-1] + pd.Timedelta(hours=12))
ax.set_xticks(nov.index[::4])
ax.grid(True, axis="y", linestyle="--", alpha=0.4)
fig.text(0.01, 0.01, src + ". Shows when orders were placed, not why.", fontsize=8)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(OUT / "03_orders_per_day_nov2017.png", dpi=200); plt.close(fig)

# 04 item price per month
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.plot(xs, W["item_price_R$"], marker="o", linewidth=2, color="#2ca02c")
ax.annotate(f"{mx_p}: R$ {W['item_price_R$'].max():,.0f}", (mx_p.to_timestamp(), W["item_price_R$"].max()),
            textcoords="offset points", xytext=(-10, 10), ha="right", fontsize=9,
            arrowprops=dict(arrowstyle="-", color="grey"))
nov_p = W.loc["2017-11", "item_price_R$"]
ax.annotate(f"2017-11: R$ {nov_p:,.0f}", (pd.Period("2017-11").to_timestamp(), nov_p),
            textcoords="offset points", xytext=(-10, 10), ha="right", fontsize=9,
            arrowprops=dict(arrowstyle="-", color="grey"))
ax.set_title(f"Item price per month, {W.index[0].strftime('%b %Y')} to {W.index[-1].strftime('%b %Y')} "
             f"(R$, n = {items_w:,} items)", fontsize=13, fontweight="bold")
ax.set_ylabel("Sum of item price (R$), freight excluded"); ax.set_xlabel("Purchase month")
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2)); ax.xaxis.set_major_formatter(mfmt)
ax.yaxis.set_major_formatter(thou); ax.grid(True, linestyle="--", alpha=0.4); ax.set_ylim(0, W["item_price_R$"].max() * 1.12)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
fig.text(0.01, 0.01, src + ". Each item counted once (not once per payment row).\n" + AUG_NOTE, fontsize=8)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(OUT / "04_item_price_per_month.png", dpi=200); plt.close(fig)
for f in sorted(OUT.glob("*.png")):
    say("saved", f.relative_to(BASE))

(OUT / "results_summary.txt").write_text("\n".join(LOG), encoding="utf-8")
say(f"\nsaved {(OUT / 'results_summary.txt').relative_to(BASE)}")
say("DONE")