"""
build_warehouse.py
Loads all 6 CSVs into SQLite and creates analytics tables.
Run once before starting the app.
"""
import sqlite3, pandas as pd, numpy as np, re, nltk, joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap, os

try:
    from nltk.corpus import stopwords
    STOP = set(stopwords.words("english"))
except:
    nltk.download("stopwords")
    from nltk.corpus import stopwords
    STOP = set(stopwords.words("english"))

DB = "data/warehouse.db"
print("=" * 60)
print("Building Customer Intelligence Platform Data Warehouse")
print("=" * 60)

# ── 1. LOAD RAW CSVs ─────────────────────────────────────────────
print("\n[1/7] Loading CSVs...")
users      = pd.read_csv("data/users.csv")
orders     = pd.read_csv("data/orders.csv", parse_dates=["order_date"])
items      = pd.read_csv("data/order_items.csv")
products   = pd.read_csv("data/products.csv")
reviews    = pd.read_csv("data/reviews.csv", parse_dates=["review_date"])
events     = pd.read_csv("data/events.csv", parse_dates=["event_timestamp"])

print(f"  Users: {len(users):,}  Orders: {len(orders):,}  Items: {len(items):,}")
print(f"  Products: {len(products):,}  Reviews: {len(reviews):,}  Events: {len(events):,}")

# ── 2. SENTIMENT ON REVIEWS ──────────────────────────────────────
print("\n[2/7] Running sentiment analysis on reviews...")

def clean(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return " ".join(w for w in text.split() if w not in STOP)

# Label by rating: 1-2 = negative, 3 = neutral(drop), 4-5 = positive
reviews["sentiment_label"] = reviews["rating"].map(
    {1: 0, 2: 0, 3: None, 4: 1, 5: 1}
)
labeled = reviews.dropna(subset=["sentiment_label"]).copy()
labeled["clean_text"] = labeled["review_text"].apply(clean)
labeled["sentiment_label"] = labeled["sentiment_label"].astype(int)

tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1,2))
X = tfidf.fit_transform(labeled["clean_text"])
y = labeled["sentiment_label"]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
lr = LogisticRegression(max_iter=500)
lr.fit(X_tr, y_tr)
acc = accuracy_score(y_te, lr.predict(X_te))
print(f"  Sentiment model accuracy: {acc:.3f}")

# Predict on ALL reviews
reviews["clean_text"] = reviews["review_text"].apply(clean)
all_vec = tfidf.transform(reviews["clean_text"])
reviews["predicted_sentiment"] = lr.predict(all_vec)
reviews["sentiment_confidence"] = lr.predict_proba(all_vec).max(axis=1)

# SHAP explainer
bg = X_tr[:300]
explainer = shap.LinearExplainer(lr, bg, feature_perturbation="interventional")

os.makedirs("models", exist_ok=True)
joblib.dump(lr, "models/sentiment_model.pkl")
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
joblib.dump(explainer, "models/shap_explainer.pkl")

# ── 3. RFM SEGMENTATION ──────────────────────────────────────────
print("\n[3/7] Computing RFM segmentation...")
completed = orders[orders["order_status"] == "completed"]
snapshot  = completed["order_date"].max()

rfm = completed.groupby("user_id").agg(
    recency   = ("order_date", lambda x: (snapshot - x.max()).days),
    frequency = ("order_id",   "count"),
    monetary  = ("total_amount", "sum"),
).reset_index()

# Score 1-5 per dimension
for col in ["recency", "frequency", "monetary"]:
    try:
        if col == "recency":
            rfm[f"{col}_score"] = pd.qcut(rfm[col], 5, labels=[5,4,3,2,1], duplicates="drop").astype(int)
        else:
            rfm[f"{col}_score"] = pd.qcut(rfm[col], 5, labels=[1,2,3,4,5], duplicates="drop").astype(int)
    except:
        rfm[f"{col}_score"] = 3

rfm["rfm_score"] = rfm["recency_score"] + rfm["frequency_score"] + rfm["monetary_score"]

def segment(row):
    r, f, m = row["recency_score"], row["frequency_score"], row["monetary_score"]
    if r >= 4 and f >= 4 and m >= 4: return "VIP"
    elif r >= 3 and f >= 3:           return "Loyal"
    elif r >= 3:                       return "Regular"
    elif r <= 2 and f >= 3:            return "At Risk"
    else:                              return "Lost"

rfm["segment"] = rfm.apply(segment, axis=1)
print(f"  Segments: {rfm['segment'].value_counts().to_dict()}")

# ── 4. CLV ESTIMATION ────────────────────────────────────────────
print("\n[4/7] Estimating Customer Lifetime Value...")
clv = completed.groupby("user_id").agg(
    total_spent     = ("total_amount", "sum"),
    order_count     = ("order_id", "count"),
    avg_order_value = ("total_amount", "mean"),
).reset_index()

days_active = completed.groupby("user_id")["order_date"].apply(
    lambda x: max((x.max() - x.min()).days, 1)
).reset_index(name="days_active")
clv = clv.merge(days_active, on="user_id")
clv["purchase_rate"] = clv["order_count"] / (clv["days_active"] / 30 + 1)
clv["clv_12m"] = clv["avg_order_value"] * clv["purchase_rate"] * 12
clv["clv_12m"] = clv["clv_12m"].clip(upper=clv["clv_12m"].quantile(0.99))

# ── 5. CHURN PREDICTION ──────────────────────────────────────────
print("\n[5/7] Training churn prediction model...")
snapshot_all = orders["order_date"].max()
last_order   = orders.groupby("user_id")["order_date"].max().reset_index()
last_order["days_since"] = (snapshot_all - last_order["order_date"]).dt.days
last_order["churned"]    = (last_order["days_since"] > 90).astype(int)

user_feat = orders.groupby("user_id").agg(
    order_count     = ("order_id", "count"),
    avg_order_value = ("total_amount", "mean"),
    total_spent     = ("total_amount", "sum"),
    cancelled_count = ("order_status", lambda x: (x == "cancelled").sum()),
    returned_count  = ("order_status", lambda x: (x == "returned").sum()),
).reset_index()

review_feat = reviews.groupby("user_id").agg(
    avg_rating       = ("rating", "mean"),
    review_count     = ("review_id", "count"),
    neg_review_count = ("predicted_sentiment", lambda x: (x == 0).sum()),
).reset_index()

event_feat = events.groupby("user_id").agg(
    total_events    = ("event_id", "count"),
    cart_events     = ("event_type", lambda x: (x == "cart").sum()),
    view_events     = ("event_type", lambda x: (x == "view").sum()),
    purchase_events = ("event_type", lambda x: (x == "purchase").sum()),
).reset_index()

churn_df = (last_order
    .merge(user_feat,   on="user_id", how="left")
    .merge(review_feat, on="user_id", how="left")
    .merge(event_feat,  on="user_id", how="left")
    .fillna(0))

feat_cols = ["days_since","order_count","avg_order_value","total_spent",
             "cancelled_count","returned_count","avg_rating","review_count",
             "neg_review_count","total_events","cart_events","view_events","purchase_events"]

X_c = churn_df[feat_cols]
y_c = churn_df["churned"]

X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_c, y_c, test_size=0.2, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_tr2, y_tr2)
churn_acc = accuracy_score(y_te2, rf.predict(X_te2))
print(f"  Churn model accuracy: {churn_acc:.3f}")

churn_df["churn_probability"] = rf.predict_proba(X_c)[:, 1]
churn_df["churn_risk"] = pd.cut(churn_df["churn_probability"],
    bins=[0,.33,.66,1], labels=["Low","Medium","High"])

joblib.dump(rf, "models/churn_model.pkl")

# ── 6. PRODUCT HEALTH SCORE ──────────────────────────────────────
print("\n[6/7] Computing Product Health Scores...")
prod_sales = items.merge(orders[["order_id","order_status","order_date"]], on="order_id")
prod_sales = prod_sales[prod_sales["order_status"] == "completed"]

sales_agg = prod_sales.groupby("product_id").agg(
    units_sold    = ("quantity", "sum"),
    revenue       = ("item_total", "sum"),
    order_count   = ("order_id", "nunique"),
).reset_index()

rev_agg = reviews.groupby("product_id").agg(
    avg_rating     = ("rating", "mean"),
    review_count   = ("review_id", "count"),
    neg_reviews    = ("predicted_sentiment", lambda x: (x==0).sum()),
    pos_reviews    = ("predicted_sentiment", lambda x: (x==1).sum()),
).reset_index()

health = products.merge(sales_agg, on="product_id", how="left")
health = health.merge(rev_agg, on="product_id", how="left").fillna(0)

def normalize(s):
    mn, mx = s.min(), s.max()
    return ((s - mn) / (mx - mn + 1e-9) * 100).clip(0,100)

health["sales_score"]      = normalize(health["revenue"])
health["rating_score"]     = (health["avg_rating"] / 5 * 100).clip(0,100)
health["sentiment_score"]  = np.where(
    health["review_count"] > 0,
    health["pos_reviews"] / (health["review_count"] + 1e-9) * 100, 50)
health["complaint_rate"]   = np.where(
    health["review_count"] > 0,
    health["neg_reviews"] / (health["review_count"] + 1e-9) * 100, 0)

health["health_score"] = (
    0.35 * health["sales_score"] +
    0.25 * health["rating_score"] +
    0.25 * health["sentiment_score"] -
    0.15 * health["complaint_rate"]
).clip(0, 100).round(1)

# ── 7. WRITE TO SQLITE ───────────────────────────────────────────
print("\n[7/7] Writing to SQLite warehouse...")
conn = sqlite3.connect(DB)

users.to_sql("users", conn, if_exists="replace", index=False)
orders.to_sql("orders", conn, if_exists="replace", index=False)
items.to_sql("order_items", conn, if_exists="replace", index=False)
products.to_sql("products", conn, if_exists="replace", index=False)
reviews.to_sql("reviews", conn, if_exists="replace", index=False)
events.to_sql("events", conn, if_exists="replace", index=False)
rfm.to_sql("rfm_segments", conn, if_exists="replace", index=False)
clv.to_sql("customer_clv", conn, if_exists="replace", index=False)
churn_df.to_sql("churn_predictions", conn, if_exists="replace", index=False)
health.to_sql("product_health", conn, if_exists="replace", index=False)

conn.close()

print(f"\n Warehouse built: {DB}")
print(f"   Tables: users, orders, order_items, products, reviews, events,")
print(f"           rfm_segments, customer_clv, churn_predictions, product_health")
print("\n Models saved: sentiment_model, tfidf_vectorizer, shap_explainer, churn_model")
print("\nRun: streamlit run app.py")
