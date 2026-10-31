"""
01_build_order_table.py
Turns the 113,390-ROW file into a table with ONE ROW PER DISTINCT ORDER.

Run from the repository root:
    python Model1\01_build_order_table.py
Optional: python Model1\01_build_order_table.py path/to/your_file.xlsx

Reads : Model1/data/Brazilian-Ecommerce-Cleaned.xlsx   (or the path you pass in)
Writes: Model1/data/olist_order_level.csv, Model1/outputs/cleaning_log.txt
It never changes your original file.

Everyday example: the file is like a receipt roll where one shopping basket
with 2 items and 2 payments is printed as 4 lines. Adding up the lines would
count the same money twice. So we first list each ITEM once, then add up the
items basket by basket.
"""
import sys
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent

# Updated to use your specific absolute Windows path as the default fallback
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "data" / "Brazilian-Ecommerce-Cleaned.xlsx"
OUT_CSV = BASE / "data" / "olist_order_level.csv"
OUT_LOG = BASE / "outputs" / "cleaning_log.txt"

OUT_CSV.parent.mkdir(exist_ok=True, parents=True)
OUT_LOG.parent.mkdir(exist_ok=True, parents=True)


EXPECTED_ROWS = 113_390
EXPECTED_ORDERS = 95_128

log = []
def say(msg=""):
    print(msg)
    log.append(str(msg))

def stop(msg):
    say("\nSTOPPED WITH ERROR: " + msg)
    OUT_LOG.write_text("\n".join(log), encoding="utf-8")
    sys.exit(1)

# ------------------------------------------------------------------ load
say("CLEANING LOG")
say(f"source file: {SRC.name}")
df = pd.read_csv(SRC) if SRC.suffix.lower() == ".csv" else pd.read_excel(SRC)
say(f"rows read: {len(df):,}   columns: {df.shape[1]}")
if len(df) != EXPECTED_ROWS:
    stop(f"expected {EXPECTED_ROWS:,} rows, got {len(df):,}. Wrong file?")

need = ["order_id", "order_item_id", "customer_state", "seller_id", "seller_state", "product_id",
        "price", "freight_value", "product_weight_g", "product_length_cm", "product_height_cm",
        "product_width_cm", "order_purchase_timestamp", "order_approved_at",
        "order_delivered_customer_date", "order_estimated_delivery_date", "order_status"]
missing = [c for c in need if c not in df.columns]
if missing:
    stop(f"missing columns: {missing}")

for c in ["order_purchase_timestamp", "order_approved_at",
          "order_delivered_customer_date", "order_estimated_delivery_date"]:
    df[c] = pd.to_datetime(df[c], errors="coerce")
say(f"missing values in the columns we use: {int(df[need].isna().sum().sum())}")
if df[need].isna().sum().sum() > 0:
    stop("missing values found in needed columns; decide how to treat them first")

# ------------------------------------------------------------------ step 1: one row per ITEM
KEY = ["order_id", "order_item_id"]
item_cols = ["seller_id", "seller_state", "product_id", "price", "freight_value",
             "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]
bad = [c for c in item_cols if (df.groupby(KEY)[c].nunique() > 1).any()]
if bad:
    stop(f"these item columns change between the payment-copies of one item: {bad}")
say("check passed: item values (price, freight, weight, seller, product) are identical across the payment-copies of an item")

items = df.drop_duplicates(KEY).copy()
say(f"rows -> distinct items (order_id + order_item_id): {len(df):,} -> {len(items):,}  "
    f"({len(df) - len(items):,} copy-rows created by several payments, removed)")

# ------------------------------------------------------------------ step 2: order-level facts that must not change inside an order
order_cols = ["customer_state", "order_purchase_timestamp", "order_approved_at",
              "order_delivered_customer_date", "order_estimated_delivery_date", "order_status"]
bad = [c for c in order_cols if (df.groupby("order_id")[c].nunique() > 1).any()]
if bad:
    stop(f"these order-level columns change inside one order: {bad}")
say("check passed: customer state, all dates and status are the same on every row of an order")

# ------------------------------------------------------------------ step 3: add up items, basket by basket
items["volume_cm3"] = items["product_length_cm"] * items["product_height_cm"] * items["product_width_cm"]
items["cross"] = (items["seller_state"] != items["customer_state"]).astype(int)

agg = items.groupby("order_id").agg(
    n_items=("order_item_id", "size"),
    n_sellers=("seller_id", "nunique"),
    price_total=("price", "sum"),
    freight_total=("freight_value", "sum"),
    weight_total_g=("product_weight_g", "sum"),
    volume_total_cm3=("volume_cm3", "sum"),
    any_cross_state=("cross", "max"),
)
first = df.drop_duplicates("order_id").set_index("order_id")[order_cols]
orders = first.join(agg)

# ------------------------------------------------------------------ step 4: features known at approval time + the target
p = orders["order_purchase_timestamp"]; a = orders["order_approved_at"]
e = orders["order_estimated_delivery_date"]; d = orders["order_delivered_customer_date"]
orders["est_window_days"] = (e - p).dt.total_seconds() / 86400      # promise shown to the customer at purchase
orders["approval_hours"] = (a - p).dt.total_seconds() / 3600       # known the moment the order is approved
orders["purchase_dow"] = p.dt.dayofweek                             # 0 = Monday
orders["purchase_month"] = p.dt.to_period("M").astype(str)          # for the month table only, NOT a model feature
orders["is_late"] = (d > e).astype(int)                             # TARGET: delivered after the estimate, exact time
orders = orders.reset_index()

# ------------------------------------------------------------------ the stop-with-error checks
if not orders["order_id"].is_unique:
    stop("order_id is NOT unique in the order table")
if len(orders) != EXPECTED_ORDERS:
    stop(f"expected {EXPECTED_ORDERS:,} orders, got {len(orders):,}")
if (orders["approval_hours"] < 0).any():
    stop("some orders were approved before they were purchased")
if (orders["est_window_days"] <= 0).any():
    stop("some estimated dates are not after the purchase date")
if orders["n_items"].sum() != len(items):
    stop("items do not add up to the item table")
say(f"\nCHECKS PASSED: order_id unique, row count = {len(orders):,}, item counts add up, no negative times")

# ------------------------------------------------------------------ what the cleaning did to the money
say("\nMONEY: counted once per item vs summed over all rows (this is the inflation avoided)")
for col, name in [("price", "price"), ("freight_value", "freight")]:
    once, raw = items[col].sum(), df[col].sum()
    say(f"  {name:8s} once per item {once:>14,.2f} | over all rows {raw:>14,.2f} | inflation {raw / once - 1:.1%}")
say(f"  order table price_total sum = {orders['price_total'].sum():,.2f} (matches once-per-item: "
    f"{abs(orders['price_total'].sum() - items['price'].sum()) < 0.01})")

# ------------------------------------------------------------------ things we looked at but did NOT change
say("\nLEFT AS THEY ARE (flagged, not changed):")
say(f"  canceled orders: {int((orders['order_status'] == 'canceled').sum())} "
    f"(they still carry a delivery date; kept so the count stays 95,128; late among them: "
    f"{int(orders.loc[orders['order_status'] == 'canceled', 'is_late'].sum())})")
say(f"  items with product_weight_g == 0: {int((items['product_weight_g'] == 0).sum())} (kept as 0)")
say(f"  orders with freight_total == 0: {int((orders['freight_total'] == 0).sum()):,}")
say(f"  orders with more than 1 seller: {int((orders['n_sellers'] > 1).sum()):,}  (kept, flagged by n_sellers)")
say(f"  orders with any item from a different state than the customer: {int(orders['any_cross_state'].sum()):,}")
say(f"  payment columns: not carried into the order table (Model 1 does not use payment data)")
say(f"  seller state: used only to build any_cross_state, not kept as a feature (an order can have several sellers, so one order has no single seller state)")

# ------------------------------------------------------------------ target
n_late = int(orders["is_late"].sum())
say(f"\nTARGET is_late (delivered > estimated, exact timestamp): {n_late:,} of {len(orders):,} = {n_late / len(orders):.2%}")

# ------------------------------------------------------------------ column roles (so nobody uses the wrong ones as features)
FEATURES = ["customer_state", "purchase_dow", "est_window_days", "approval_hours", "n_items", "n_sellers",
            "any_cross_state", "price_total", "freight_total", "weight_total_g", "volume_total_cm3"]
say("\nCOLUMN ROLES in data/olist_order_level.csv")
say("  id          : order_id")
say("  FEATURES    : " + ", ".join(FEATURES))
say("  target      : is_late")
say("  OUTCOME-ONLY, never a feature: order_delivered_customer_date (the estimated date IS known at purchase; it enters only as est_window_days)")
say("  tables-only : order_status, purchase_month, order_purchase_timestamp, order_approved_at")
say("  NOT in file : carrier handover date, payment columns, seller_state, seller_id")

keep = (["order_id"] + FEATURES + ["is_late", "order_status", "purchase_month",
        "order_purchase_timestamp", "order_approved_at",
        "order_estimated_delivery_date", "order_delivered_customer_date"])
orders[keep].to_csv(OUT_CSV, index=False)
say(f"\nsaved {OUT_CSV.relative_to(BASE)}  ({len(orders):,} rows x {len(keep)} columns)")
OUT_LOG.write_text("\n".join(log), encoding="utf-8")
