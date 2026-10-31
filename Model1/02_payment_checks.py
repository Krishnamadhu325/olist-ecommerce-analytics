"""
02_payment_checks.py   READ-ONLY. Looks at the odd payment rows and asks:
"do they matter for Model 1?"

Run:  python 02_payment_checks.py        (needs data/olist_order_level.csv from step 01)
Reads: data/Brazilian-Ecommerce-Cleaned.xlsx (or a path you pass in) and data/olist_order_level.csv
Writes: outputs/payment_checks.txt, outputs/flagged_payment_orders.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "data" / "Brazilian-Ecommerce-Cleaned.xlsx"
orders = pd.read_csv(BASE / "data" / "olist_order_level.csv")
df = pd.read_csv(SRC) if SRC.suffix.lower() == ".csv" else pd.read_excel(SRC)

lines = []
def say(m=""):
    print(m); lines.append(str(m))

late = orders.set_index("order_id")["is_late"]
base_rate = late.mean()
say(f"overall: {len(orders):,} orders, late rate {base_rate:.2%}\n")

def late_line(label, ids):
    ids = pd.Index(ids).unique()
    n = len(ids); k = int(late.reindex(ids).sum())
    return f"{label}: {n} orders, {k} late ({k / n:.1%})" if n else f"{label}: 0 orders"

# ------------------------------------------------------------------ A. payment_value == 0
say("=" * 70 + "\nA. ROWS WITH payment_value == 0\n" + "=" * 70)
z = df[df["payment_value"] == 0]
cols = ["order_id", "order_item_id", "payment_sequential", "payment_type", "payment_installments",
        "price", "freight_value", "payment_value", "order_status"]
zz = z[cols].copy(); zz["order_id"] = zz["order_id"].str[:8] + "..."
say(zz.to_string(index=False))
zids = z["order_id"].unique()
say(f"\ndistinct orders involved: {len(zids)}")
for oid in zids:
    o = df[df["order_id"] == oid].drop_duplicates(["order_id", "payment_sequential"])
    paid = o["payment_value"].sum()
    it = df[df["order_id"] == oid].drop_duplicates(["order_id", "order_item_id"])
    say(f"  {oid[:8]}...: payments {list(zip(o['payment_sequential'], o['payment_type'], o['payment_value']))} "
        f"| items price+freight = {(it['price'] + it['freight_value']).sum():.2f} | paid total = {paid:.2f}")
say(late_line("late among these orders", zids))

# ------------------------------------------------------------------ B. payment_installments == 0
say("\n" + "=" * 70 + "\nB. ROWS WITH payment_installments == 0\n" + "=" * 70)
i0 = df[df["payment_installments"] == 0]
zz = i0[cols].copy(); zz["order_id"] = zz["order_id"].str[:8] + "..."
say(zz.to_string(index=False))
iids = i0["order_id"].unique()
say(f"\ndistinct orders involved: {len(iids)}")
say(late_line("late among these orders", iids))

# ------------------------------------------------------------------ C. paid total vs price + freight
say("\n" + "=" * 70 + "\nC. ORDERS WHERE THE PAID TOTAL DOES NOT MATCH price + freight\n" + "=" * 70)
items = df.drop_duplicates(["order_id", "order_item_id"])
pays = df.drop_duplicates(["order_id", "payment_sequential"])
it_tot = items.groupby("order_id").apply(lambda g: (g["price"] + g["freight_value"]).sum(), include_groups=False).rename("items_total")
pay_tot = pays.groupby("order_id")["payment_value"].sum().rename("paid_total")
c = pd.concat([it_tot, pay_tot], axis=1).dropna()
c["diff"] = c["paid_total"] - c["items_total"]
say(f"orders compared: {len(c):,}")
say(f"match within 0.05: {(c['diff'].abs() <= 0.05).sum():,} ({(c['diff'].abs() <= 0.05).mean():.2%})")
big = c[c["diff"].abs() > 1.00]
say(f"differ by more than 1.00: {len(big)} ({len(big) / len(c):.2%})")
say(f"  paid MORE than items: {(big['diff'] > 0).sum()} | paid LESS than items: {(big['diff'] < 0).sum()}")
say(f"  size of gap in R$ (absolute): median {big['diff'].abs().median():.2f}, "
    f"90th pct {big['diff'].abs().quantile(.9):.2f}, max {big['diff'].abs().max():.2f}")

# why? check installments and voucher
extra = pays.groupby("order_id").agg(max_inst=("payment_installments", "max"),
                                     has_voucher=("payment_type", lambda s: (s == "voucher").any()),
                                     n_pay=("payment_sequential", "nunique"))
b = big.join(extra)
over = b[b["diff"] > 0]; under = b[b["diff"] < 0]
if len(over):
    say(f"  of the 'paid more' orders, share with 2+ installments: {(over['max_inst'] > 1).mean():.0%} "
        f"(interest on installments would explain this)")
if len(under):
    say(f"  of the 'paid less' orders, share containing a voucher: {under['has_voucher'].mean():.0%}")
say(f"  share of flagged orders with 2+ payments: {(b['n_pay'] > 1).mean():.0%}")

bids = big.index
say("\n" + late_line("late among the flagged 302-type orders", bids))
rest = late.drop(bids, errors="ignore")
say(f"late rate in all other orders: {rest.mean():.2%} ({int(rest.sum()):,} of {len(rest):,})")

# ------------------------------------------------------------------ D. does any of it touch Model 1?
say("\n" + "=" * 70 + "\nD. DOES ANY OF THIS TOUCH MODEL 1?\n" + "=" * 70)
flagged = pd.Index(zids).union(pd.Index(iids)).union(pd.Index(bids))
say(f"all flagged orders together: {len(flagged)} of {len(orders):,} ({len(flagged) / len(orders):.2%})")
say(f"payment columns used in Model 1: none (decision 3)")
say(f"price_total / freight_total in Model 1 come from the item rows, not from payment_value")
fl = orders[orders["order_id"].isin(flagged)]
rest = orders[~orders["order_id"].isin(flagged)]
for col in ["price_total", "freight_total", "est_window_days", "n_items"]:
    say(f"  mean {col:16s} flagged {fl[col].mean():9.2f} | others {rest[col].mean():9.2f}")
pd.DataFrame({"order_id": flagged}).to_csv(BASE / "outputs" / "flagged_payment_orders.csv", index=False)

# canceled orders: size the effect
say("\n" + "=" * 70 + "\nE. CANCELED ORDERS (found while checking)\n" + "=" * 70)
can = orders[orders["order_status"] == "canceled"]
say(f"canceled orders in the 95,128: {len(can)}; late among them: {int(can['is_late'].sum())}")
say(f"late rate with them: {orders['is_late'].mean():.4%}; without them: {orders.loc[orders['order_status'] != 'canceled', 'is_late'].mean():.4%}")

(BASE / "outputs" / "payment_checks.txt").write_text("\n".join(lines), encoding="utf-8")
