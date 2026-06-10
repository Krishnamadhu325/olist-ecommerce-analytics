# Olist E-Commerce Analytics System

A 5-model end-to-end analytics project built on the [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/enzoschitini/brazilian-e-commerce-public-dataset-by-olist) (113,390 orders, 2016–2018) as part of my data analytics internship at **Voleergo Solutions LLP**.

The project covers the full analytics workflow: raw data → Excel & Power Query (null validation, dimensionality reduction) → Python modelling → Power BI dashboard.

---

## Dashboard Preview

| Product Analytics | Customer Analytics |
|---|---|
| ![Product Analytics](Power%20BI/Product_Analytics.png) | ![Customer Analytics](Power%20BI/Customer_Analytics.png) |

---

## Project Structure

```
Olist_E-commerce_Project/
│
├── Notebooks/
│   ├── module1_delivery_delay.ipynb
│   ├── module2_geographic.ipynb
│   ├── module3_classifier.ipynb
│   ├── module4_timeseries.ipynb
│   └── module5_segmentation.ipynb
│
├── Outputs/
│   ├── Module1/
│   ├── Module2/
│   ├── Module3/
│   │   └── feature_importance.png
│   ├── Module4/
│   │   └── time_series_decomposition.png
│   └── Module5/
│
├── Power BI/
│   ├── Product_Analytics.png
│   └── Customer_Analytics.png
│
├── .gitignore
└── README.md
```

---

## Models Overview

| Module | Task | Algorithm | Key Metric |
|--------|------|-----------|------------|
| 1 | Delivery Delay Prediction | Random Forest Regressor | MAE: 8.38 days, R²: 0.14 |
| 2 | Geographic Logistics Analysis | Aggregation + Mapping | — |
| 3 | Product Category Classification | Random Forest Classifier | 73 categories, Top-20 confusion matrix |
| 4 | Order Volume Time Series | Seasonal Decomposition (statsmodels) | period=6 |
| 5 | Customer Payment Segmentation | K-Means Clustering | Silhouette scoring (10K sample) |

---

## Module Details

### Module 1 — Delivery Delay Predictor
Predicts how many days late an order will arrive using seller, product, and freight features.

**Key decision:** Removed `delivery_days` and `shipping_time` as input features after identifying data leakage — these variables are calculated *from* the delivery outcome, not before it. Fixing this gave a more honest and generalisable model.

### Module 2 — Geographic Logistics Analysis
Aggregates seller concentration and customer distribution by Brazilian state. Identified that SP (São Paulo) dominates order volume, while northeastern states show higher average delivery delays due to shipping distance.

### Module 3 — Product Category Classifier
Classifies products into 73 categories using physical and pricing features.

**Key finding:** Physical dimensions (length, height, width) were stronger predictors of product category than price — product size encodes category information more reliably than cost.

**Key decision:** Used `GroupShuffleSplit` on `product_id` to prevent the same product appearing in both train and test sets (leakage prevention).

### Module 4 — Time Series Decomposition
Decomposes monthly order volume (Oct 2016 – Aug 2018) into trend, seasonality, and residual components using `statsmodels` seasonal decomposition.

**Key finding:** Strong upward growth trend throughout the period, with a clear seasonal pattern repeating every 6 months. `period=6` was selected based on the 22-month dataset span.

### Module 5 — Customer Payment Segmentation
Segments customers by payment behaviour (payment type, installments, order value) using K-Means clustering.

**Key decision:** Silhouette scoring was run on a 10,000-record random sample to keep computation feasible while maintaining statistical reliability.

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Data processing | Python, pandas, NumPy, Power Query, Excel |
| Machine learning | scikit-learn (RandomForestRegressor, RandomForestClassifier, KMeans) |
| Time series | statsmodels |
| Visualisation | matplotlib, seaborn, Power BI |
| Dashboard | Power BI Desktop (DAX measures, custom navy/amber theme) |

---

## How to Run

1. Download the Olist dataset from [Kaggle](https://www.kaggle.com/datasets/enzoschitini/brazilian-e-commerce-public-dataset-by-olist) and place CSV files in a `data/` folder.
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn statsmodels matplotlib seaborn
   ```
3. Run each module script independently in order (Module 1 → 5). Output plots save to `Outputs/Module{N}/`.
4. Open `Olist_Dashboard.pbix` in Power BI Desktop to explore the dashboard.

---

## Key Learnings

- Data leakage is subtle — features that look useful can encode the answer you're trying to predict. Catching this in Module 1 and Module 3 changed the model design significantly.
- `GroupShuffleSplit` is essential when your dataset has repeated entities (products, sellers) — random splits cause optimistic evaluation.
- Power BI DAX measures allow dynamic calculations that static Python outputs can't match for business reporting.

---

## Dataset

- **Source:** [Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/enzoschitini/brazilian-e-commerce-public-dataset-by-olist) via Kaggle
- **Records:** 113,390 orders
- **Period:** October 2016 – August 2018
- **License:** CC BY-NC-SA 4.0

---

## Author

**Krishna K M**  
BCA Graduate, Yenepoya Institute of Arts, Science, Commerce and Management (2026)    

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/krishna-km-aa3534292)
[![GitHub](https://img.shields.io/badge/GitHub-Profile-black)](https://github.com/Krishnamadhu325)
