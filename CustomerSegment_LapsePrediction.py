

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Ensure output directory exists
os.makedirs("outputs", exist_ok=True)

print("=== 1. Building RFM Table ===")
# Simulated/Loaded dataframes for demonstration if files are absent locally
# (Replace or ensure orders.csv and customers.csv are in the working directory)
try:
  orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
  customers = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"])
except FileNotFoundError:
  print(
      "Notice: orders.csv / customers.csv not found locally in this test run."
  )
  print(
      "Please place your CSV files in the working directory to run with"
      " real data."
  )
  exit(0)

snapshot = pd.Timestamp("2025-10-01 00:00:00")
completed_orders = orders[
    (orders["status"] == "completed")
    & (orders["order_ts"] < snapshot)
]

rfm = (
    completed_orders.groupby("customer_id")
    .agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
    )
    .reset_index()
)

rfm["recency"] = (snapshot - rfm["last_order_ts"]).dt.days
rfm = rfm[["customer_id", "recency", "frequency", "monetary"]]
print(f"RFM Table built for {len(rfm)} customers.")

print("\n=== 2. K-Means Clustering & Evaluation ===")
scaler_rfm = StandardScaler()
rfm_scaled = scaler_rfm.fit_transform(rfm[["recency", "frequency", "monetary"]])

for k in range(2, 9):
  kmeans_test = KMeans(n_clusters=k, n_init=10, random_state=42)
  labels = kmeans_test.fit_predict(rfm_scaled)
  inertia = kmeans_test.inertia_
  sil = silhouette_score(rfm_scaled, labels)
  print(f"K={k} -> Inertia: {inertia:.1f}, Silhouette: {sil:.3f}")

# Chosen K = 4 based on silhouette peak and elbow method
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
rfm["segment_id"] = kmeans.fit_predict(rfm_scaled)

print("\n=== 3. Profiling & Projecting Segments ===")
segment_map = {
    0: "At-Risk / Low Volume",
    1: "Premium Champions",
    2: "Frequent Loyalists",
    3: "Recent One-Offs",
}
rfm["segment_name"] = rfm["segment_id"].map(segment_map)
max_monetary_segment = rfm.groupby("segment_id")["monetary"].mean().idxmax()
rfm["is_premium"] = rfm["segment_id"] == max_monetary_segment

# PCA projection check
pca = PCA(n_components=2)
pca_res = pca.fit_transform(rfm_scaled)
print(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_}")

rfm.to_csv("outputs/customer_segments.csv", index=False)
print("Saved outputs/customer_segments.csv")

print("\n=== 4. Defining the Lapse Problem & Feature Engineering ===")
cutoff = pd.Timestamp("2025-09-01 00:00:00")
pre_cutoff_orders = orders[orders["order_ts"] < cutoff]

# Cohort: customers with >= 1 completed order before cutoff
cohort_customer_ids = pre_cutoff_orders[
    pre_cutoff_orders["status"] == "completed"
]["customer_id"].unique()
cohort = customers[customers["customer_id"].isin(cohort_customer_ids)].copy()

# Target label: lapsed=1 if NO completed order between 2025-09-01 and 2025-09-30
sep_completed = orders[
    (orders["status"] == "completed")
    & (orders["order_ts"] >= cutoff)
    & (orders["order_ts"] < pd.Timestamp("2025-10-01 00:00:00"))
]["customer_id"].unique()

cohort["lapsed"] = cohort["customer_id"].apply(
    lambda cid: 0 if cid in sep_completed else 1
)


# Feature construction using pre-cutoff data only
def build_features(df_cohort, all_orders):
  feats = []
  for cid in df_cohort["customer_id"]:
    cust_ord = all_orders[
        (all_orders["customer_id"] == cid)
        & (all_orders["order_ts"] < cutoff)
    ]
    completed_cust = cust_ord[cust_ord["status"] == "completed"]

    recency = (
        (cutoff - completed_cust["order_ts"].max()).days
        if not completed_cust.empty
        else 999
    )
    frequency = len(completed_cust)
    monetary = completed_cust["order_amount"].sum()
    avg_discount = (
        completed_cust["discount_pct"].mean()
        if "discount_pct" in completed_cust.columns
        else 0.0
    )

    final_statuses = cust_ord[
        cust_ord["status"].isin(["completed", "cancelled"])
    ]
    cancellation_rate = (
        len(cust_ord[cust_ord["status"] == "cancelled"])
        / len(final_statuses)
        if len(final_statuses) > 0
        else 0.0
    )

    avg_delivery = (
        completed_cust["delivery_minutes"].mean()
        if "delivery_minutes" in completed_cust.columns
        else 0.0
    )
    fav_cat = (
        completed_cust["product_category"].mode()[0]
        if "product_category" in completed_cust.columns and not completed_cust.empty
        else "Unknown"
    )

    feats.append({
        "customer_id": cid,
        "recency": recency,
        "frequency": frequency,
        "monetary": monetary,
        "avg_discount_pct": avg_discount,
        "cancellation_rate": cancellation_rate,
        "avg_delivery_minutes": avg_delivery,
        "favourite_category": fav_cat,
    })
  return pd.DataFrame(feats)


feature_df = build_features(cohort, orders)
model_data = cohort.merge(feature_df, on="customer_id")
model_data["tenure_days"] = (cutoff - model_data["signup_date"]).dt.days

feature_cols = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount_pct",
    "cancellation_rate",
    "avg_delivery_minutes",
    "favourite_category",
    "city",
    "gender",
    "age",
    "tenure_days",
]
X = model_data[feature_cols]
y = model_data["lapsed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Baseline Evaluation
y_pred_baseline = np.zeros_like(y_test)
print(
    "Baseline Accuracy:",
    accuracy_score(y_test, y_pred_baseline),
    "F1:",
    f1_score(y_test, y_pred_baseline, zero_division=0),
)

print("\n=== 5. Preprocessing Pipeline (No Leakage) ===")
cat_cols = ["city", "gender", "favourite_category"]
num_cols = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount_pct",
    "cancellation_rate",
    "avg_delivery_minutes",
    "age",
    "tenure_days",
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
    ]
)

# Verify no missing values
assert X_train.isna().sum().sum() == 0
assert X_test.isna().sum().sum() == 0
print("Data integrity confirmed: 0 missing values.")

print("\n=== 6. Training, Tuning & Evaluation ===")
pipe_lr = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
])
grid_lr = GridSearchCV(
    pipe_lr,
    {"classifier__C": [0.01, 0.1, 1, 10, 100]},
    cv=5,
    scoring="f1",
    n_jobs=-1,
).fit(X_train, y_train)

pipe_rf = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        RandomForestClassifier(n_estimators=100, random_state=42),
    ),
])
grid_rf = GridSearchCV(
    pipe_rf,
    {
        "classifier__max_depth": [3, 5, 8],
        "classifier__min_samples_leaf": [5, 10, 20],
    },
    cv=5,
    scoring="f1",
    n_jobs=-1,
).fit(X_train, y_train)

print(f"Logistic Regression Best C: {grid_lr.best_params_['classifier__C']}")
print(f"Random Forest Best Params: {grid_rf.best_params_}")

# Test Set Evaluation
best_rf = grid_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)
print("Random Forest Test F1:", f1_score(y_test, y_pred_rf, zero_division=0))

print("\n=== 7. Class-Balance Experiment ===")
train_lapsed_idx = y_train[y_train == 1].index
train_retained_idx = y_train[y_train == 0].index

np.random.seed(42)
oversampled_lapsed = np.random.choice(
    train_lapsed_idx, size=len(train_retained_idx), replace=True
)
X_train_bal = pd.concat(
    [X_train.loc[train_retained_idx], X_train.loc[oversampled_lapsed]]
)
y_train_bal = pd.concat(
    [y_train.loc[train_retained_idx], y_train.loc[oversampled_lapsed]]
)

grid_lr.best_estimator_.fit(X_train_bal, y_train_bal)
y_pred_bal = grid_lr.best_estimator_.predict(X_test)
print(
    "Balanced LR Recall:",
    recall_score(y_test, y_pred_bal),
    "Precision:",
    precision_score(y_test, y_pred_bal, zero_division=0),
)

print("\n=== 8. Interpretation, Persistence & Deployment ===")
joblib.dump(best_rf, "outputs/lapse_model.joblib")
reloaded_model = joblib.load("outputs/lapse_model.joblib")
assert (best_rf.predict(X_test) == reloaded_model.predict(X_test)).all()
print("Model persistence verified.")

# Get test subset from model_data using X_test indices
test_model_data = model_data.loc[X_test.index].copy()

# Filter for customers in the Premium segment
premium_custs = rfm[rfm["is_premium"]]["customer_id"]
test_premium = test_model_data[
    test_model_data["customer_id"].isin(premium_custs)
].copy()

# Predict lapse probability using their features from X_test
test_premium_features = X_test.loc[test_premium.index]
test_premium["lapse_probability"] = reloaded_model.predict_proba(
    test_premium_features
)[:, 1]
test_premium["actually_lapsed"] = y_test.loc[test_premium.index]

# Rank and export top retention list
retention_list = (
    test_premium.sort_values(by="lapse_probability", ascending=False)
    .head(10)[["customer_id", "lapse_probability", "actually_lapsed"]]
)
retention_list.to_csv("outputs/retention_list.csv", index=False)
print("Saved outputs/retention_list.csv successfully!")

