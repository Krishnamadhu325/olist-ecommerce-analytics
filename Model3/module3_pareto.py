"""
Olist Module 3: product category revenue concentration (Pareto) at ITEM grain.

Folder layout (run from anywhere, paths are relative to this file):
    module3_pareto.py
    data/Brazilian-Ecommerce-Cleaned.xlsx
    data/product_category_name_translation.csv
    outputs/module3/          <- created by this script (PNG + CSV + results_summary.txt)

Units used on purpose, never mixed:
    rows  = items x payments (113,390)          -> only used to REPLAY the old method
    items = order_id + order_item_id (108,640)  -> the grain for money (Pareto)
"""
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------------------------------------------------------------
# 0. SETTINGS
# ----------------------------------------------------------------------------
try:
    BASE = Path(__file__).resolve().parent
except NameError:                      # running inside a notebook
    BASE = Path.cwd()
DATA = BASE / "data"
OUT = BASE / "outputs" / "module3"
OUT.mkdir(parents=True, exist_ok=True)

XLSX = DATA / "Brazilian-Ecommerce-Cleaned.xlsx"
TRANS = DATA / "product_category_name_translation.csv"

# Data-version checks (they stop you running the wrong file)
EXPECTED_ROWS, EXPECTED_ITEMS, EXPECTED_PRODUCTS, EXPECTED_CATEGORIES = 113_390, 108_640, 31_625, 73

ADDED_ENGLISH = {
    "pc_gamer": "pc_gamer",
    "portateis_cozinha_e_preparadores_de_alimentos": "portable_kitchen_food_preparers",
}
# Deliberate spelling / meaning fixes applied on top of the file (original kept in translation_used.csv)
FIX_ENGLISH = {
    "costruction_tools_garden": "construction_tools_garden",
    "costruction_tools_tools": "construction_tools_tools",
    "fashio_female_clothing": "fashion_female_clothing",
    "home_confort": "home_comfort",
    "arts_and_craftmanship": "arts_and_craftsmanship",
    "security_and_services": "insurance_and_services",
}

summary_lines = []


def say(text=""):
    print(text)
    summary_lines.append(str(text))


def head(title):
    say("\n" + "=" * 78)
    say(title)
    say("=" * 78)


say(f"Python {sys.version.split()[0]} | pandas {pd.__version__} | numpy {np.__version__} | "
    f"matplotlib {matplotlib.__version__}")

# ----------------------------------------------------------------------------
# 1. LOAD + GRAIN ASSERTS
# ----------------------------------------------------------------------------
head("STEP 1  Load data and check the grain")
df = pd.read_excel(XLSX)
say(f"rows: {len(df):,} | columns: {df.shape[1]}")
assert len(df) == EXPECTED_ROWS, f"expected {EXPECTED_ROWS:,} rows, got {len(df):,}: wrong file?"

ITEM_KEY = ["order_id", "order_item_id"]
items = df.drop_duplicates(ITEM_KEY).copy()
# ITEM-GRAIN ASSERT: one row per item, and price/freight do not change between the payment rows of an item
assert items["order_unique_id"].is_unique
check = df.groupby(ITEM_KEY)[["price", "freight_value"]].nunique()
assert (check <= 1).all().all(), "price or freight differs between payment rows of one item"
assert len(items) == EXPECTED_ITEMS, f"expected {EXPECTED_ITEMS:,} items, got {len(items):,}"
say(f"items (order_id + order_item_id): {len(items):,}  -> one row per item, assert passed")
say(f"distinct orders: {df['order_id'].nunique():,} | distinct products: {df['product_id'].nunique():,} "
    f"| distinct categories: {df['product_category_name'].nunique()}")
say(f"order_status (rows): {df['order_status'].value_counts().to_dict()}  (canceled orders are kept)")
assert df["product_category_name"].notna().all()
assert df["product_id"].nunique() == EXPECTED_PRODUCTS
n_cat = df["product_category_name"].nunique()
assert n_cat == EXPECTED_CATEGORIES, f"expected {EXPECTED_CATEGORIES} categories, got {n_cat}"

# ----------------------------------------------------------------------------
# 2. ENGLISH CATEGORY NAMES
# ----------------------------------------------------------------------------
head("STEP 2  English category names")
tr = pd.read_csv(TRANS, encoding="utf-8-sig")        # utf-8-sig: the file starts with a hidden BOM character
tr.columns = ["pt", "en_in_file"]
assert tr["pt"].is_unique
rows_t = []
for pt, en in zip(tr["pt"], tr["en_in_file"]):
    used = FIX_ENGLISH.get(en, en)
    rows_t.append((pt, en, used, "typo/meaning fixed" if used != en else "file"))
for pt, en in ADDED_ENGLISH.items():
    assert pt not in set(tr["pt"])
    used = FIX_ENGLISH.get(en, en)
    rows_t.append((pt, "", used, "added (not in the translation file)"))
tmap = pd.DataFrame(rows_t, columns=["pt", "en_in_file", "en_used", "source"])
assert tmap["pt"].is_unique and tmap["en_used"].is_unique, "English names must be one-to-one"
missing = set(df["product_category_name"]) - set(tmap["pt"])
assert not missing, f"categories without an English name: {missing}"
EN = dict(zip(tmap["pt"], tmap["en_used"]))
tmap.to_csv(OUT / "translation_used.csv", index=False)
say(f"translation file rows: {len(tr)} | added: {len(ADDED_ENGLISH)} | "
    f"names changed on top of the file: {(tmap['source'] == 'typo/meaning fixed').sum()}")
say(tmap[tmap["source"] != "file"][["pt", "en_in_file", "en_used", "source"]].to_string(index=False))
items["category_en"] = items["product_category_name"].map(EN)
df["category_en"] = df["product_category_name"].map(EN)
assert items["category_en"].notna().all()

# ----------------------------------------------------------------------------
# 3. DISTINCT PRODUCTS PER CATEGORY (context column for the Pareto table)
# ----------------------------------------------------------------------------
head("STEP 3  Distinct products per category")
assert items.groupby("product_id")["category_en"].nunique().max() == 1, "a product sits in two categories"
prod_cat = items.drop_duplicates("product_id")
assert len(prod_cat) == EXPECTED_PRODUCTS
cat_products = prod_cat["category_en"].value_counts()
say(f"products: {len(prod_cat):,} | categories: {len(cat_products)} | "
    f"largest by products: {cat_products.index[0]} ({cat_products.iloc[0]:,})")

# ----------------------------------------------------------------------------
# 4. PARETO AT ITEM GRAIN
# ----------------------------------------------------------------------------
head("STEP 4  Revenue Pareto at ITEM grain")
items["price_plus_freight"] = items["price"] + items["freight_value"]
cat_tbl = (items.groupby("category_en")
           .agg(n_items=("order_id", "size"), price=("price", "sum"),
                freight=("freight_value", "sum"), price_plus_freight=("price_plus_freight", "sum"))
           .join(cat_products.rename("n_products")).reset_index())
assert len(cat_tbl) == n_cat
assert abs(cat_tbl["price"].sum() - items["price"].sum()) < 0.01


def pareto(col):
    t = cat_tbl.sort_values(col, ascending=False).reset_index(drop=True)
    t["cum_pct"] = t[col].cumsum() / t[col].sum() * 100
    n80 = int((t["cum_pct"] < 80).sum() + 1)
    return t, n80


par_price, n80_price = pareto("price")
par_pf, n80_pf = pareto("price_plus_freight")
say(f"Totals at item grain: price {items['price'].sum():,.2f} | freight {items['freight_value'].sum():,.2f} "
    f"| price + freight {items['price_plus_freight'].sum():,.2f}")
pay = df.drop_duplicates(["order_id", "payment_sequential"])
say(f"Other possible revenue totals (for reference): payment_value over distinct payments "
    f"{pay['payment_value'].sum():,.2f} | payment_value over all rows {df['payment_value'].sum():,.2f}")
for name, t, n80, col in [("PRICE ONLY (headline)", par_price, n80_price, "price"), 
                          ("PRICE + FREIGHT", par_pf, n80_pf, "price_plus_freight")]:
    say(f"\n{name}: {n_cat} categories | 80% reached at {n80} categories "
        f"(cumulative at {n80 - 1} = {t.loc[n80 - 2, 'cum_pct']:.2f}%, at {n80} = {t.loc[n80 - 1, 'cum_pct']:.2f}%)")
    say("  top-5 cumulative %: " + " / ".join(f"{v:.2f}" for v in t["cum_pct"].head(5)))
    say("  top-5: " + ", ".join(t["category_en"].head(5)))
    say(f"  top-10 cumulative %: {t['cum_pct'].iloc[9]:.2f}")
    top10_items = int(t["n_items"].head(10).sum())
    n80_items = int(t["n_items"].head(n80).sum())
    say(f"  items in the top 10 categories: {top10_items:,} of {len(items):,} ({top10_items / len(items) * 100:.1f}%)")
    say(f"  items in the first {n80} categories: {n80_items:,} of {len(items):,} ({n80_items / len(items) * 100:.1f}%)")
    say(f"  category that crosses 80%: {t.loc[n80 - 1, 'category_en']}")
par_price.to_csv(OUT / "pareto_price_items.csv", index=False)
par_pf.to_csv(OUT / "pareto_price_plus_freight_items.csv", index=False)
only_price = set(par_price["category_en"].head(n80_price)) - set(par_pf["category_en"].head(n80_pf))
only_pf = set(par_pf["category_en"].head(n80_pf)) - set(par_price["category_en"].head(n80_price))
say(f"\nCount to reach 80%: price {n80_price} vs price + freight {n80_pf}. "
    f"Only in the price-only list: {sorted(only_price) or 'none'} | only in the price + freight list: {sorted(only_pf) or 'none'}")

# how much does the top-20 list depend on the unit?
t_rows = set(df["category_en"].value_counts().head(20).index)
t_items = set(items["category_en"].value_counts().head(20).index)
t_prod = set(cat_products.head(20).index)
say(f"Top-20 lists (swapped categories): rows vs items {sorted(t_rows - t_items)} -> {sorted(t_items - t_rows)}; "
    f"items vs products {sorted(t_items - t_prod)} -> {sorted(t_prod - t_items)}")

# ----------------------------------------------------------------------------
# 4b. GRAIN CHECKS (support the README statements about rows vs items)
# ----------------------------------------------------------------------------
head("STEP 4b  Grain checks")
cats_per_order = items.groupby("order_id")["category_en"].nunique()
say(f"orders with items from more than one category: {(cats_per_order > 1).sum():,} of {len(cats_per_order):,}")
ipo = items.groupby("order_id").size()
ppo = pay.groupby("order_id").size()
exp_rows = int((ipo * ppo.reindex(ipo.index)).sum())
say(f"rows expected if every item repeats once per payment: {exp_rows:,} (actual {len(df):,}) "
    f"-> {'MATCH' if exp_rows == len(df) else 'DIFFERENT'}")

# ----------------------------------------------------------------------------
# 5. OLD METHOD REPLAY (price + freight summed over ROWS) + OLD vs NEW
# ----------------------------------------------------------------------------
head("STEP 5  Replay of the old row-level Pareto, and old vs new")
df["revenue_old"] = df["price"] + df["freight_value"]
old_rev = df.groupby("category_en")["revenue_old"].sum().sort_values(ascending=False)
old_cum = old_rev.cumsum() / old_rev.sum() * 100
old_n80 = int((old_cum < 80).sum() + 1)


def ok(flag):
    return "OK" if flag else "DIFFERENT"


say(f"replay check - old top-5 cumulative 39.58: {old_cum.iloc[4]:.2f} {ok(abs(old_cum.iloc[4] - 39.58) < 0.01)}")
say(f"(old method reaches 80% at {old_n80} categories - the earlier version did not report this)")
over_price = (df["price"].sum() / items["price"].sum() - 1) * 100
over_freight = (df["freight_value"].sum() / items["freight_value"].sum() - 1) * 100
say(f"summing over rows overstates total price by {over_price:.2f}% and freight by {over_freight:.2f}%")

old_new = pd.DataFrame([
    ("Money unit", f"rows ({len(df):,})", f"items ({len(items):,})"),
    ("Total price", f"{df['price'].sum():,.2f}", f"{items['price'].sum():,.2f}"),
    ("Total freight", f"{df['freight_value'].sum():,.2f}", f"{items['freight_value'].sum():,.2f}"),
    ("Revenue definition", "price + freight", "price only (price + freight as a check)"),
    ("Top-5 cumulative % (old: price+freight)", f"{old_cum.iloc[4]:.2f}",
     f"{par_price['cum_pct'].iloc[4]:.2f} price | {par_pf['cum_pct'].iloc[4]:.2f} price+freight"),
    ("Categories needed for 80%", f"{old_n80} (not printed)",
     f"{n80_price} of {n_cat} (price) | {n80_pf} of {n_cat} (price+freight)"),
], columns=["item", "old", "new"])
old_new.to_csv(OUT / "old_vs_new.csv", index=False)
say("\nOLD vs NEW:")
say(old_new.to_string(index=False))

# ----------------------------------------------------------------------------
# 6. CHARTS (PNG, relative paths)
# ----------------------------------------------------------------------------
head("STEP 6  Charts")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
NAVY, AMBER, GREY, RED = "#1f3a5f", "#e8a317", "#b8bec6", "#c0392b"

# 6.1 Pareto (bars = item price on the left axis; line = cumulative % on the right axis)
t = par_price
TOP_LABELS = 20                                   # only the first 20 names are printed (73 names are unreadable)
cum_prev = t.loc[n80_price - 2, "cum_pct"]
cum_n80 = t.loc[n80_price - 1, "cum_pct"]
fig, ax1 = plt.subplots(figsize=(16, 7.5))
ax1.bar(range(len(t)), t["price"] / 1e6, color=[NAVY if i < n80_price else GREY for i in range(len(t))])
ax1.set_xticks(range(len(t)))
ax1.set_xticklabels([c if i < TOP_LABELS else "" for i, c in enumerate(t["category_en"])],
                    rotation=60, ha="right", fontsize=9)
ax1.set_xlabel(f"Product categories ranked by item price (names shown for the top {TOP_LABELS} of {n_cat}; "
               f"full list in pareto_price_items.csv)")
ax1.set_ylabel("Item price summed over items (R$ million)")
ax2 = ax1.twinx()
ax2.spines["right"].set_visible(True)
ax2.plot(range(len(t)), t["cum_pct"], color=AMBER, lw=2, marker=".", ms=4)
# the 80% guides stop where they meet the curve, so the horizontal line cannot be misread against the left axis
ax2.hlines(80, -1, n80_price - 0.5, color=RED, ls="--", lw=1.2)
ax2.vlines(n80_price - 0.5, 0, 80, color=RED, ls="--", lw=1.2)
ax2.set_ylim(0, 105)
ax2.set_yticks([0, 20, 40, 60, 80, 100])
ax2.set_ylabel("Cumulative share of item price (%)")
ax1.set_xlim(-1, len(t))
ax2.annotate(f"80% is crossed at category {n80_price} of {n_cat} ({t.loc[n80_price - 1, 'category_en']})\n"
             f"{n80_price - 1} categories = {cum_prev:.1f}%, {n80_price} categories = {cum_n80:.1f}%\n"
             f"(price + freight also crosses 80% at category {n80_pf})",
             xy=(n80_price - 0.5, 80), xytext=(min(n80_price + 6, len(t) - 28), 55), fontsize=11, color=RED,
             arrowprops=dict(arrowstyle="->", color=RED))
ax1.set_title(f"Pareto of item price by product category (n = {len(items):,} items, {n_cat} categories; freight excluded)",
              fontsize=13, loc="left")
fig.tight_layout()
fig.savefig(OUT / "01_pareto_items.png", dpi=200)
plt.close(fig)

# 6.2 Top 10 categories by item price (horizontal bars, labelled)
top10 = par_price.head(10).iloc[::-1]             # reversed so the largest bar is at the top
share = top10["price"] / items["price"].sum() * 100
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(top10["category_en"], share, color=NAVY)
for b, (_, r) in zip(bars, top10.iterrows()):
    ax.text(b.get_width() + 0.15, b.get_y() + b.get_height() / 2,
            f"R$ {r['price'] / 1e6:.2f}M | {int(r['n_items']):,} items", va="center", fontsize=9)
ax.set_xlim(0, share.max() * 1.45)
ax.set_xlabel("Share of total item price (%)")
ax.set_title(f"Top 10 categories by item price = {par_price['cum_pct'].iloc[9]:.1f}% of the total "
             f"(R$ {items['price'].sum() / 1e6:.2f}M, {len(items):,} items)", fontsize=12, loc="left")
fig.tight_layout()
fig.savefig(OUT / "02_top10_categories.png", dpi=200)
plt.close(fig)
say("saved: " + ", ".join(sorted(p.name for p in OUT.glob("*.png"))))

# ----------------------------------------------------------------------------
# 7. SAVE THE PASTE-BACK SUMMARY
# ----------------------------------------------------------------------------
(OUT / "results_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
print(f"\nDone. Everything is in {OUT}")