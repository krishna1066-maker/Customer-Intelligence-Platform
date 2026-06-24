"""
Customer Intelligence Platform v3
Real E-Commerce Data — 9 Modules
"""
import os, sys, re, sqlite3, joblib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
try:
    from nltk.corpus import stopwords
    STOP = set(stopwords.words("english"))
except:
    import nltk; nltk.download("stopwords")
    from nltk.corpus import stopwords
    STOP = set(stopwords.words("english"))

# ── CONFIG ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="🛒", layout="wide",
    initial_sidebar_state="expanded"
)

DB = "data/warehouse.db"

# ── CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
.kpi-card {
    background: linear-gradient(135deg,#667eea,#764ba2);
    border-radius:12px; padding:18px 20px;
    color:white; text-align:center; margin-bottom:8px;
}
.kpi-card h2{font-size:2rem;margin:0;font-weight:800;}
.kpi-card p {margin:0;opacity:.85;font-size:.82rem;letter-spacing:.5px;}
.green-card {background:linear-gradient(135deg,#11998e,#38ef7d);}
.red-card   {background:linear-gradient(135deg,#c0392b,#e74c3c);}
.blue-card  {background:linear-gradient(135deg,#2980b9,#6dd5fa);}
.orange-card{background:linear-gradient(135deg,#f7971e,#ffd200);}
.seg-vip    {background:#6c5ce7;color:white;padding:4px 10px;border-radius:20px;font-size:.8rem;}
.seg-loyal  {background:#00b894;color:white;padding:4px 10px;border-radius:20px;font-size:.8rem;}
.seg-reg    {background:#0984e3;color:white;padding:4px 10px;border-radius:20px;font-size:.8rem;}
.seg-risk   {background:#e17055;color:white;padding:4px 10px;border-radius:20px;font-size:.8rem;}
.seg-lost   {background:#636e72;color:white;padding:4px 10px;border-radius:20px;font-size:.8rem;}
</style>
""", unsafe_allow_html=True)

# ── DB HELPER ─────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def q(sql): 
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query(sql, conn)
    conn.close()
    return df

def kpi(label, value, cls=""):
    st.markdown(f'<div class="kpi-card {cls}"><h2>{value}</h2><p>{label}</p></div>',
                unsafe_allow_html=True)

# ── MODELS ────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    return (joblib.load("models/sentiment_model.pkl"),
            joblib.load("models/tfidf_vectorizer.pkl"),
            joblib.load("models/shap_explainer.pkl"),
            joblib.load("models/churn_model.pkl"))

sentiment_model, tfidf, shap_explainer, churn_model = load_models()

def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", str(text).lower())
    return " ".join(w for w in text.split() if w not in STOP)

# ── SIDEBAR ───────────────────────────────────────────────────────
PAGES = [
    "🏠 Executive Dashboard",
    "👥 Customer Segmentation",
    "💰 Customer Lifetime Value",
    "⚠️ Churn Prediction",
    "🏥 Product Health Engine",
    "⭐ Review Intelligence",
    "🎯 Recommendation Engine",
    "📊 Behavioral Analytics",
    "💡 PM Insights",
]

with st.sidebar:
    st.markdown("## 🛒 CIP v3")
    st.markdown("**Customer Intelligence Platform**")
    st.markdown("*Real E-Commerce Data · 9 Modules*")
    st.markdown("---")
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    st.markdown("---")
    st.caption("📦 10,000 Users · 20,000 Orders")
    st.caption("📝 15,000 Reviews · 80,000 Events")
    st.caption("🛍️ 2,000 Products · 43,525 Items")

# ══════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE DASHBOARD
# ══════════════════════════════════════════════════════════════════
if page == PAGES[0]:
    st.title("🏠 Executive Dashboard")
    st.caption("Real-time KPIs from 20,000 orders · 10,000 customers · 2,000 products")
    st.markdown("---")

    rev   = q("SELECT SUM(total_amount) as r FROM orders WHERE order_status='completed'")["r"][0]
    users_n = q("SELECT COUNT(DISTINCT user_id) as n FROM orders")["n"][0]
    ord_n   = q("SELECT COUNT(*) as n FROM orders WHERE order_status='completed'")["n"][0]
    aov     = q("SELECT AVG(total_amount) as a FROM orders WHERE order_status='completed'")["a"][0]
    repeat  = q("""SELECT ROUND(100.0*COUNT(*)/SUM(COUNT(*)) OVER(),1) as pct
                   FROM (SELECT user_id,COUNT(*) as c FROM orders WHERE order_status='completed'
                         GROUP BY user_id) WHERE c>1""")["pct"][0]
    cancelled = q("SELECT ROUND(100.0*SUM(CASE WHEN order_status='cancelled' THEN 1 ELSE 0 END)/COUNT(*),1) as p FROM orders")["p"][0]

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1: kpi("Total Revenue", f"${rev:,.0f}", "green-card")
    with c2: kpi("Total Customers", f"{users_n:,}")
    with c3: kpi("Completed Orders", f"{ord_n:,}", "blue-card")
    with c4: kpi("Avg Order Value", f"${aov:,.1f}", "orange-card")
    with c5: kpi("Repeat Purchase %", f"{repeat}%", "green-card")
    with c6: kpi("Cancellation %", f"{cancelled}%", "red-card")

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("📈 Monthly Revenue Trend")
        rev_trend = q("""
            SELECT strftime('%Y-%m', order_date) as month,
                   SUM(total_amount) as revenue,
                   COUNT(*) as orders
            FROM orders WHERE order_status='completed'
            GROUP BY month ORDER BY month
        """)
        fig = px.area(rev_trend, x="month", y="revenue",
                      color_discrete_sequence=["#667eea"],
                      labels={"revenue":"Revenue ($)","month":"Month"})
        fig.update_layout(height=320, margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("🏷️ Revenue by Category")
        cat_rev = q("""
            SELECT p.category, SUM(oi.item_total) as revenue
            FROM order_items oi
            JOIN products p ON oi.product_id=p.product_id
            JOIN orders o ON oi.order_id=o.order_id
            WHERE o.order_status='completed'
            GROUP BY p.category ORDER BY revenue DESC
        """)
        fig2 = px.bar(cat_rev, x="category", y="revenue",
                      color="revenue", color_continuous_scale="Viridis",
                      labels={"revenue":"Revenue ($)","category":"Category"})
        fig2.update_layout(height=320, margin=dict(t=10,b=10), showlegend=False,
                           coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("📦 Order Status Breakdown")
        status_df = q("SELECT order_status, COUNT(*) as count FROM orders GROUP BY order_status")
        fig3 = px.pie(status_df, names="order_status", values="count",
                      color_discrete_sequence=px.colors.qualitative.Set3, hole=0.4)
        fig3.update_layout(height=300, margin=dict(t=10,b=10))
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        st.subheader("⭐ Rating Distribution")
        rat_df = q("SELECT rating, COUNT(*) as count FROM reviews GROUP BY rating ORDER BY rating")
        colors = ["#e74c3c","#e67e22","#f1c40f","#2ecc71","#27ae60"]
        fig4 = px.bar(rat_df, x="rating", y="count",
                      color="rating", color_discrete_sequence=colors,
                      labels={"count":"Reviews","rating":"Rating"})
        fig4.update_layout(height=300, showlegend=False, margin=dict(t=10,b=10))
        st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 2 — CUSTOMER SEGMENTATION (RFM)
# ══════════════════════════════════════════════════════════════════
elif page == PAGES[1]:
    st.title("👥 Customer Segmentation — RFM Analysis")
    st.caption("Recency · Frequency · Monetary segmentation on completed orders")
    st.markdown("---")

    rfm = q("SELECT * FROM rfm_segments")

    seg_counts = rfm["segment"].value_counts().reset_index()
    seg_counts.columns = ["segment","count"]

    seg_colors = {"VIP":"#6c5ce7","Loyal":"#00b894","Regular":"#0984e3",
                  "At Risk":"#e17055","Lost":"#636e72"}

    c1,c2,c3,c4,c5 = st.columns(5)
    for col, seg in zip([c1,c2,c3,c4,c5], ["VIP","Loyal","Regular","At Risk","Lost"]):
        n = rfm[rfm["segment"]==seg].shape[0]
        with col: kpi(f"{seg} Customers", f"{n:,}")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🥧 Segment Distribution")
        fig = px.pie(seg_counts, names="segment", values="count",
                     color="segment",
                     color_discrete_map=seg_colors, hole=0.4)
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("💰 Avg Monetary Value by Segment")
        mon_seg = rfm.groupby("segment")["monetary"].mean().reset_index()
        mon_seg.columns = ["segment","avg_spend"]
        mon_seg = mon_seg.sort_values("avg_spend", ascending=False)
        fig2 = px.bar(mon_seg, x="segment", y="avg_spend",
                      color="segment", color_discrete_map=seg_colors,
                      labels={"avg_spend":"Avg Total Spend ($)"},
                      text="avg_spend")
        fig2.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        fig2.update_layout(height=360, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 RFM Score Distribution")
    col3,col4,col5 = st.columns(3)
    with col3:
        fig3 = px.histogram(rfm, x="recency", nbins=30, title="Recency (days since last order)",
                            color_discrete_sequence=["#e17055"])
        fig3.update_layout(height=280, margin=dict(t=30,b=10))
        st.plotly_chart(fig3, use_container_width=True)
    with col4:
        fig4 = px.histogram(rfm, x="frequency", nbins=20, title="Frequency (orders)",
                            color_discrete_sequence=["#00b894"])
        fig4.update_layout(height=280, margin=dict(t=30,b=10))
        st.plotly_chart(fig4, use_container_width=True)
    with col5:
        fig5 = px.histogram(rfm, x="monetary", nbins=30, title="Monetary (total spend $)",
                            color_discrete_sequence=["#6c5ce7"])
        fig5.update_layout(height=280, margin=dict(t=30,b=10))
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown("---")
    st.subheader("🔍 Customer Lookup")
    uid = st.text_input("Enter User ID (e.g. U000001)")
    if uid:
        row = rfm[rfm["user_id"]==uid]
        if not row.empty:
            r = row.iloc[0]
            st.success(f"**Segment:** {r['segment']}  |  **RFM Score:** {r['rfm_score']}  |  "
                       f"**Recency:** {r['recency']} days  |  **Frequency:** {int(r['frequency'])} orders  |  "
                       f"**Monetary:** ${r['monetary']:,.2f}")
        else:
            st.warning("User not found in completed orders.")


# ══════════════════════════════════════════════════════════════════
# PAGE 3 — CUSTOMER LIFETIME VALUE
# ══════════════════════════════════════════════════════════════════
elif page == PAGES[2]:
    st.title("💰 Customer Lifetime Value")
    st.caption("Projected 12-month revenue per customer based on purchase velocity")
    st.markdown("---")

    clv = q("SELECT * FROM customer_clv ORDER BY clv_12m DESC")

    c1,c2,c3,c4 = st.columns(4)
    with c1: kpi("Avg CLV (12m)", f"${clv['clv_12m'].mean():,.0f}", "green-card")
    with c2: kpi("Top CLV", f"${clv['clv_12m'].max():,.0f}", "orange-card")
    with c3: kpi("Customers Modeled", f"{len(clv):,}", "blue-card")
    with c4: kpi("Total Projected Rev", f"${clv['clv_12m'].sum():,.0f}", "green-card")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏆 Top 20 Highest-Value Customers")
        top20 = clv.head(20)
        fig = px.bar(top20, x="user_id", y="clv_12m",
                     color="clv_12m", color_continuous_scale="Greens",
                     labels={"clv_12m":"12m CLV ($)","user_id":"Customer"})
        fig.update_layout(height=380, showlegend=False, coloraxis_showscale=False,
                          xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📊 CLV Distribution")
        fig2 = px.histogram(clv, x="clv_12m", nbins=40,
                            color_discrete_sequence=["#00b894"],
                            labels={"clv_12m":"Projected 12m CLV ($)"})
        fig2.update_layout(height=380, margin=dict(t=10,b=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("📋 CLV Table — Top 50 Customers")
    display = clv.head(50)[["user_id","total_spent","order_count","avg_order_value","clv_12m"]].copy()
    display.columns = ["User ID","Total Spent","Orders","Avg Order","CLV 12m"]
    for col in ["Total Spent","Avg Order","CLV 12m"]:
        display[col] = display[col].apply(lambda x: f"${x:,.2f}")
    st.dataframe(display, use_container_width=True, height=400)


# ══════════════════════════════════════════════════════════════════
# PAGE 4 — CHURN PREDICTION
# ══════════════════════════════════════════════════════════════════
elif page == PAGES[3]:
    st.title("⚠️ Churn Prediction")
    st.caption("Random Forest model · Features: recency, frequency, spend, reviews, events")
    st.markdown("---")

    churn = q("SELECT * FROM churn_predictions")

    high   = (churn["churn_risk"]=="High").sum()
    medium = (churn["churn_risk"]=="Medium").sum()
    low    = (churn["churn_risk"]=="Low").sum()
    churned_n = churn["churned"].sum()

    c1,c2,c3,c4 = st.columns(4)
    with c1: kpi("High Risk Customers", f"{high:,}", "red-card")
    with c2: kpi("Medium Risk", f"{medium:,}", "orange-card")
    with c3: kpi("Low Risk", f"{low:,}", "green-card")
    with c4: kpi("Already Churned (90d)", f"{churned_n:,}", "red-card")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🥧 Churn Risk Distribution")
        risk_df = churn["churn_risk"].value_counts().reset_index()
        risk_df.columns = ["risk","count"]
        fig = px.pie(risk_df, names="risk", values="count", hole=0.4,
                     color="risk",
                     color_discrete_map={"High":"#e74c3c","Medium":"#e67e22","Low":"#27ae60"})
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📈 Churn Probability Distribution")
        fig2 = px.histogram(churn, x="churn_probability", nbins=40,
                            color_discrete_sequence=["#e74c3c"],
                            labels={"churn_probability":"Churn Probability"})
        fig2.update_layout(height=340, margin=dict(t=10,b=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("🚨 High-Risk Customers — Immediate Action Required")
    high_risk = churn[churn["churn_risk"]=="High"].sort_values(
        "churn_probability", ascending=False).head(20)
    hr_display = high_risk[["user_id","churn_probability","days_since",
                              "order_count","total_spent","avg_rating"]].copy()
    hr_display.columns = ["User ID","Churn Prob","Days Inactive","Orders","Total Spent","Avg Rating"]
    hr_display["Churn Prob"] = hr_display["Churn Prob"].apply(lambda x: f"{x:.1%}")
    hr_display["Total Spent"] = hr_display["Total Spent"].apply(lambda x: f"${x:,.0f}")
    hr_display["Avg Rating"]  = hr_display["Avg Rating"].apply(lambda x: f"{x:.1f}")
    st.dataframe(hr_display, use_container_width=True, height=380)


# ══════════════════════════════════════════════════════════════════
# PAGE 5 — PRODUCT HEALTH ENGINE
# ══════════════════════════════════════════════════════════════════
elif page == PAGES[4]:
    st.title("🏥 Product Health Engine")
    st.caption("Health Score = Sales (35%) + Rating (25%) + Sentiment (25%) − Complaints (15%)")
    st.markdown("---")

    ph = q("SELECT * FROM product_health ORDER BY health_score DESC")
    cats = ["All"] + sorted(ph["category"].unique().tolist())
    sel_cat = st.selectbox("Filter Category", cats)
    if sel_cat != "All":
        ph = ph[ph["category"]==sel_cat]

    avg_health = ph["health_score"].mean()
    best = ph.iloc[0]["product_name"] if len(ph) else "-"
    worst = ph.iloc[-1]["product_name"] if len(ph) else "-"
    critical = (ph["health_score"] < 30).sum()

    c1,c2,c3,c4 = st.columns(4)
    with c1: kpi("Avg Health Score", f"{avg_health:.1f}/100")
    with c2: kpi("Healthiest Product", best[:20] + "...", "green-card")
    with c3: kpi("Critical Products", f"{critical}", "red-card")
    with c4: kpi("Products Analyzed", f"{len(ph):,}", "blue-card")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏆 Top 15 Healthiest Products")
        top15 = ph.head(15)
        fig = px.bar(top15, x="health_score", y="product_name",
                     orientation="h", color="health_score",
                     color_continuous_scale=["#e74c3c","#f39c12","#27ae60"],
                     range_color=[0,100],
                     labels={"health_score":"Health Score","product_name":"Product"},
                     text="health_score")
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig.update_layout(height=480, yaxis=dict(autorange="reversed"),
                          showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🚨 Bottom 15 — Needs Attention")
        bot15 = ph.tail(15).sort_values("health_score")
        fig2 = px.bar(bot15, x="health_score", y="product_name",
                      orientation="h", color="health_score",
                      color_continuous_scale=["#e74c3c","#f39c12","#27ae60"],
                      range_color=[0,100],
                      labels={"health_score":"Health Score","product_name":"Product"},
                      text="health_score")
        fig2.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig2.update_layout(height=480, yaxis=dict(autorange="reversed"),
                           showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Health Score by Category")
    cat_health = q("""SELECT category,
                             ROUND(AVG(health_score),1) as avg_health,
                             COUNT(*) as products,
                             SUM(CASE WHEN health_score<30 THEN 1 ELSE 0 END) as critical
                      FROM product_health GROUP BY category ORDER BY avg_health DESC""")
    fig3 = px.bar(cat_health, x="category", y="avg_health",
                  color="avg_health", color_continuous_scale=["#e74c3c","#f39c12","#27ae60"],
                  range_color=[0,100], text="avg_health",
                  labels={"avg_health":"Avg Health Score","category":"Category"})
    fig3.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig3.update_layout(height=340, coloraxis_showscale=False)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Full Product Health Table")
    ph_display = ph[["product_name","category","brand","health_score",
                      "avg_rating","revenue","units_sold","complaint_rate"]].copy()
    ph_display.columns = ["Product","Category","Brand","Health Score",
                          "Avg Rating","Revenue","Units Sold","Complaint Rate %"]
    ph_display["Revenue"] = ph_display["Revenue"].apply(lambda x: f"${x:,.0f}")
    ph_display["Complaint Rate %"] = ph_display["Complaint Rate %"].apply(lambda x: f"{x:.1f}%")
    ph_display["Health Score"] = ph_display["Health Score"].apply(lambda x: f"{x:.1f}")
    st.dataframe(ph_display, use_container_width=True, height=420)


# ══════════════════════════════════════════════════════════════════
# PAGE 6 — REVIEW INTELLIGENCE
# ══════════════════════════════════════════════════════════════════
elif page == PAGES[5]:
    st.title("⭐ Review Intelligence")
    st.caption("Sentiment analysis · Aspect detection · SHAP explainability · Complaint extraction")
    st.markdown("---")

    ASPECTS = {
        "Quality":     ["quality","build","material","durable","sturdy","cheap","flimsy"],
        "Delivery":    ["delivery","shipping","arrived","package","late","fast","delayed"],
        "Price/Value": ["price","value","expensive","cheap","worth","overpriced","affordable"],
        "Performance": ["performance","speed","fast","slow","lag","works","failed"],
        "Support":     ["support","service","customer","refund","return","response","help"],
        "Design":      ["design","look","color","appearance","size","weight","style"],
    }
    POS_W = {"excellent","amazing","great","good","best","perfect","love","outstanding",
             "superb","brilliant","fantastic","smooth","crisp","impressive","solid","top"}
    NEG_W = {"bad","terrible","poor","worst","broken","slow","issue","problem","defect",
             "disappointed","horrible","awful","useless","waste","fail","refund","return"}

    def aspect_sent(text):
        tl = text.lower()
        words = set(re.sub(r"[^a-zA-Z\s]","",tl).split())
        out = {}
        for asp, kws in ASPECTS.items():
            if any(k in tl for k in kws):
                p = len(words & POS_W); n = len(words & NEG_W)
                out[asp] = "Negative" if n>p else ("Positive" if p>0 else "Neutral")
        return out

    COMPLAINT_P = {
        "🔥 Quality Issue":   ["quality issue","poor quality","bad quality","cheap quality"],
        "🚚 Delivery Issue":  ["late delivery","delayed","not delivered","wrong item"],
        "💸 Refund Issue":    ["refund","return","money back","replacement"],
        "⚡ Performance":     ["slow","lag","freeze","crash","doesn't work","not working"],
        "📞 Support Issue":   ["customer service","no response","support","unhelpful"],
        "🎨 Wrong Item":      ["wrong color","different from","not as described","color was"],
    }
    def detect_complaints(text):
        tl = text.lower()
        return [k for k,pats in COMPLAINT_P.items() if any(p in tl for p in pats)]

    # Overview stats
    rev_stats = q("""SELECT COUNT(*) as total,
                            SUM(CASE WHEN predicted_sentiment=1 THEN 1 ELSE 0 END) as pos,
                            SUM(CASE WHEN predicted_sentiment=0 THEN 1 ELSE 0 END) as neg,
                            ROUND(AVG(rating),2) as avg_rating
                     FROM reviews""")
    total_r = rev_stats["total"][0]
    pos_r   = rev_stats["pos"][0]
    neg_r   = rev_stats["neg"][0]
    avg_r   = rev_stats["avg_rating"][0]

    c1,c2,c3,c4 = st.columns(4)
    with c1: kpi("Total Reviews", f"{total_r:,}")
    with c2: kpi("Positive", f"{pos_r:,}", "green-card")
    with c3: kpi("Negative", f"{neg_r:,}", "red-card")
    with c4: kpi("Avg Rating", f"{avg_r:.2f} ⭐")

    st.markdown("---")

    # Live analyzer
    st.subheader("🔍 Live Review Analyzer")
    ex_col1, ex_col2, ex_col3 = st.columns(3)
    with ex_col1:
        if st.button("😊 Positive Example"):
            st.session_state["rev_input"] = "Excellent quality! Delivery was super fast. Great value for money, highly recommend this product to everyone."
    with ex_col2:
        if st.button("😡 Negative Example"):
            st.session_state["rev_input"] = "Terrible product. Color was completely different from images. Delivery was late and customer service gave no response to my refund request."
    with ex_col3:
        if st.button("😐 Mixed Example"):
            st.session_state["rev_input"] = "Performance is good and design looks nice. However delivery was delayed and the price feels a bit expensive for the quality."

    review_input = st.text_area("Enter Review Text",
                                value=st.session_state.get("rev_input",""),
                                height=130, placeholder="Type or paste a customer review...")

    if st.button("🚀 Analyze", type="primary", use_container_width=True) and review_input.strip():
        cleaned = clean_text(review_input)
        vec     = tfidf.transform([cleaned])
        pred    = sentiment_model.predict(vec)[0]
        probs   = sentiment_model.predict_proba(vec)[0]
        conf    = max(probs)*100
        aspects = aspect_sent(review_input)
        complaints = detect_complaints(review_input)

        # SHAP
        try:
            import shap as shap_lib
            shap_vals = shap_explainer.shap_values(vec)
            if isinstance(shap_vals, list): vals = shap_vals[1][0]
            else: vals = shap_vals[0]
            fn = tfidf.get_feature_names_out()
            nz = vec.nonzero()[1]
            shap_pairs = sorted([(fn[i], float(vals[i])) for i in nz],
                                 key=lambda x: abs(x[1]), reverse=True)[:8]
        except: shap_pairs = []

        st.markdown("---")
        if pred==1: st.success(f"✅ Positive Review — {conf:.1f}% confidence")
        else:       st.error(f"❌ Negative Review — {conf:.1f}% confidence")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.subheader("🎯 Aspect Sentiment")
            if aspects:
                for asp, sent in aspects.items():
                    icon = "✅" if sent=="Positive" else ("❌" if sent=="Negative" else "⚪")
                    color = "#d4edda" if sent=="Positive" else ("#f8d7da" if sent=="Negative" else "#fff3cd")
                    border = "#28a745" if sent=="Positive" else ("#dc3545" if sent=="Negative" else "#ffc107")
                    st.markdown(f'<div style="background:{color};border-left:4px solid {border};padding:8px 12px;border-radius:6px;margin:3px 0">{icon} <b>{asp}</b> — {sent}</div>',
                                unsafe_allow_html=True)
            else: st.info("No aspects detected.")

        with col_b:
            st.subheader("⚠️ Complaints")
            if complaints:
                for c in complaints: st.error(f"• {c}")
            else: st.success("✅ No complaints detected.")

        with col_c:
            st.subheader("🧠 SHAP — Why?")
            if shap_pairs:
                words_s = [p[0] for p in shap_pairs]
                vals_s  = [p[1] for p in shap_pairs]
                colors_s= ["#28a745" if v>0 else "#dc3545" for v in vals_s]
                fig_sh  = go.Figure(go.Bar(x=vals_s, y=words_s, orientation="h",
                                           marker_color=colors_s,
                                           text=[f"{v:+.3f}" for v in vals_s],
                                           textposition="outside"))
                fig_sh.update_layout(height=280, margin=dict(t=5,b=5,l=5,r=50),
                                     yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_sh, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Sentiment by Product Category")
    sent_cat = q("""
        SELECT p.category,
               ROUND(100.0*SUM(CASE WHEN r.predicted_sentiment=1 THEN 1 ELSE 0 END)/COUNT(*),1) as positive_pct,
               COUNT(*) as reviews
        FROM reviews r JOIN products p ON r.product_id=p.product_id
        GROUP BY p.category ORDER BY positive_pct DESC
    """)
    fig_sc = px.bar(sent_cat, x="category", y="positive_pct",
                    color="positive_pct",
                    color_continuous_scale=["#e74c3c","#f39c12","#27ae60"],
                    range_color=[40,90], text="positive_pct",
                    labels={"positive_pct":"Positive %","category":"Category"})
    fig_sc.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_sc.update_layout(height=340, coloraxis_showscale=False)
    st.plotly_chart(fig_sc, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 7 — RECOMMENDATION ENGINE
# ══════════════════════════════════════════════════════════════════
elif page == PAGES[6]:
    st.title("🎯 Recommendation Engine")
    st.caption("Frequently Bought Together · Cross-Sell · User-Based recommendations")
    st.markdown("---")

    # Frequently bought together (co-occurrence)
    @st.cache_data
    def get_cooccurrence():
        pairs = q("""
            SELECT a.product_id as p1, b.product_id as p2, COUNT(*) as co_count
            FROM order_items a JOIN order_items b
              ON a.order_id=b.order_id AND a.product_id < b.product_id
            GROUP BY a.product_id, b.product_id
            HAVING co_count >= 3
            ORDER BY co_count DESC
            LIMIT 500
        """)
        return pairs

    @st.cache_data
    def get_user_products(uid):
        return q(f"""
            SELECT DISTINCT oi.product_id, p.product_name, p.category, p.brand, p.price
            FROM order_items oi
            JOIN products p ON oi.product_id=p.product_id
            JOIN orders o ON oi.order_id=o.order_id
            WHERE oi.user_id='{uid}' AND o.order_status='completed'
        """)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔗 Frequently Bought Together")
        all_prods = q("SELECT product_id, product_name, category FROM products")
        prod_map  = dict(zip(all_prods["product_id"], all_prods["product_name"]))
        cat_map   = dict(zip(all_prods["product_id"], all_prods["category"]))

        sel_prod = st.selectbox("Select a Product",
                                all_prods.apply(lambda r: f"{r['product_id']} — {r['product_name']}", axis=1).tolist()[:200])
        pid = sel_prod.split(" — ")[0]
        pairs = get_cooccurrence()
        recs = pairs[(pairs["p1"]==pid)|(pairs["p2"]==pid)].copy()
        recs["other"] = recs.apply(lambda r: r["p2"] if r["p1"]==pid else r["p1"], axis=1)
        recs["product_name"] = recs["other"].map(prod_map)
        recs["category"]     = recs["other"].map(cat_map)
        recs = recs[["product_name","category","co_count"]].head(8)
        if not recs.empty:
            recs.columns = ["Recommended Product","Category","Bought Together Count"]
            st.dataframe(recs, use_container_width=True, height=300)
        else:
            st.info("Not enough co-purchase data for this product.")

    with col2:
        st.subheader("👤 User-Based Recommendations")
        uid_input = st.text_input("Enter User ID", value="U000001")
        if uid_input:
            user_prods = get_user_products(uid_input)
            if not user_prods.empty:
                st.write(f"**{len(user_prods)} products purchased by {uid_input}:**")
                st.dataframe(user_prods[["product_name","category","price"]].rename(
                    columns={"product_name":"Product","category":"Category","price":"Price"}
                ), use_container_width=True, height=180)

                # Recommend from same categories not yet bought
                cats_bought = user_prods["category"].unique().tolist()
                pids_bought = user_prods["product_id"].tolist()
                cats_str = "','".join(cats_bought)
                pids_str = "','".join(pids_bought)
                recs2 = q(f"""
                    SELECT p.product_name, p.category, p.brand, p.price, p.rating
                    FROM products p
                    WHERE p.category IN ('{cats_str}')
                      AND p.product_id NOT IN ('{pids_str}')
                    ORDER BY p.rating DESC LIMIT 8
                """)
                if not recs2.empty:
                    st.write("**Recommended (top-rated, same categories):**")
                    recs2["price"] = recs2["price"].apply(lambda x: f"${x:.2f}")
                    recs2["rating"] = recs2["rating"].apply(lambda x: f"⭐{x:.1f}")
                    st.dataframe(recs2.rename(columns={
                        "product_name":"Product","category":"Category",
                        "brand":"Brand","price":"Price","rating":"Rating"
                    }), use_container_width=True, height=280)
            else:
                st.warning("No completed orders found for this user.")

    st.markdown("---")
    st.subheader("🏆 Top Co-Purchased Product Pairs")
    top_pairs = get_cooccurrence().head(15)
    top_pairs["Product A"] = top_pairs["p1"].map(prod_map)
    top_pairs["Product B"] = top_pairs["p2"].map(prod_map)
    top_pairs = top_pairs[["Product A","Product B","co_count"]].rename(
        columns={"co_count":"Times Bought Together"})
    st.dataframe(top_pairs, use_container_width=True, height=400)


# ══════════════════════════════════════════════════════════════════
# PAGE 8 — BEHAVIORAL ANALYTICS
# ══════════════════════════════════════════════════════════════════
elif page == PAGES[7]:
    st.title("📊 Behavioral Analytics")
    st.caption("View → Cart → Purchase funnel · Abandonment analysis · Event trends")
    st.markdown("---")

    funnel = q("""
        SELECT event_type, COUNT(*) as count
        FROM events GROUP BY event_type
    """)
    fmap = dict(zip(funnel["event_type"], funnel["count"]))
    views     = fmap.get("view", 0)
    carts     = fmap.get("cart", 0)
    wishlists = fmap.get("wishlist", 0)
    purchases = fmap.get("purchase", 0)

    v2c = carts/views*100 if views else 0
    c2p = purchases/carts*100 if carts else 0
    abandon = 100 - c2p

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: kpi("Product Views", f"{views:,}", "blue-card")
    with c2: kpi("Cart Adds", f"{carts:,}", "orange-card")
    with c3: kpi("Wishlists", f"{wishlists:,}")
    with c4: kpi("Purchases", f"{purchases:,}", "green-card")
    with c5: kpi("Cart Abandonment", f"{abandon:.1f}%", "red-card")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔽 Conversion Funnel")
        funnel_df = pd.DataFrame({
            "stage": ["Views","Cart Adds","Wishlist","Purchases"],
            "count": [views, carts, wishlists, purchases],
        })
        fig = go.Figure(go.Funnel(
            y=funnel_df["stage"], x=funnel_df["count"],
            textinfo="value+percent initial",
            marker_color=["#3498db","#f39c12","#9b59b6","#27ae60"]
        ))
        fig.update_layout(height=380, margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📅 Daily Event Volume Trend")
        ev_trend = q("""
            SELECT strftime('%Y-%m', event_timestamp) as month,
                   event_type, COUNT(*) as count
            FROM events
            GROUP BY month, event_type ORDER BY month
        """)
        fig2 = px.line(ev_trend, x="month", y="count", color="event_type",
                       markers=True,
                       labels={"count":"Events","month":"Month","event_type":"Event Type"})
        fig2.update_layout(height=380)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("🏷️ Events by Category")
        cat_ev = q("""
            SELECT p.category, e.event_type, COUNT(*) as count
            FROM events e JOIN products p ON e.product_id=p.product_id
            GROUP BY p.category, e.event_type
        """)
        fig3 = px.bar(cat_ev, x="category", y="count", color="event_type",
                      barmode="group",
                      labels={"count":"Events","category":"Category","event_type":"Event"})
        fig3.update_layout(height=360, xaxis_tickangle=30)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("🛒 Top Products by Cart Adds")
        top_cart = q("""
            SELECT p.product_name, p.category, COUNT(*) as carts
            FROM events e JOIN products p ON e.product_id=p.product_id
            WHERE e.event_type='cart'
            GROUP BY e.product_id ORDER BY carts DESC LIMIT 15
        """)
        fig4 = px.bar(top_cart, x="carts", y="product_name", orientation="h",
                      color="carts", color_continuous_scale="Blues",
                      labels={"carts":"Cart Adds","product_name":"Product"})
        fig4.update_layout(height=360, yaxis=dict(autorange="reversed"),
                           coloraxis_showscale=False)
        st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 9 — PM INSIGHTS
# ══════════════════════════════════════════════════════════════════
elif page == PAGES[8]:
    st.title("💡 Product Manager Insights")
    st.caption("Auto-generated business intelligence from all data sources")
    st.markdown("---")

    # Compute insights from live data
    neg_cat = q("""
        SELECT p.category,
               ROUND(100.0*SUM(CASE WHEN r.predicted_sentiment=0 THEN 1 ELSE 0 END)/COUNT(*),1) as neg_pct
        FROM reviews r JOIN products p ON r.product_id=p.product_id
        GROUP BY p.category ORDER BY neg_pct DESC LIMIT 1
    """)
    high_churn_seg = q("""
        SELECT COUNT(*) as n FROM churn_predictions WHERE churn_probability > 0.7
    """)
    top_rev_cat = q("""
        SELECT p.category, SUM(oi.item_total) as rev
        FROM order_items oi JOIN products p ON oi.product_id=p.product_id
        JOIN orders o ON oi.order_id=o.order_id WHERE o.order_status='completed'
        GROUP BY p.category ORDER BY rev DESC LIMIT 1
    """)
    avg_abandon = q("SELECT COUNT(*) as c FROM events WHERE event_type='cart'")
    avg_purchase_ev = q("SELECT COUNT(*) as c FROM events WHERE event_type='purchase'")
    critical_products = q("SELECT COUNT(*) as n FROM product_health WHERE health_score < 30")
    top_seg_rev = q("""
        SELECT r.segment, ROUND(SUM(o.total_amount)) as revenue
        FROM rfm_segments r JOIN orders o ON r.user_id=o.user_id
        WHERE o.order_status='completed'
        GROUP BY r.segment ORDER BY revenue DESC LIMIT 1
    """)
    cancel_rate = q("SELECT ROUND(100.0*SUM(CASE WHEN order_status='cancelled' THEN 1 ELSE 0 END)/COUNT(*),1) as r FROM orders")

    insights = [
        ("🔴 High Churn Alert",
         f"**{high_churn_seg['n'][0]:,} customers** have >70% churn probability. "
         f"Immediate re-engagement campaign recommended.",
         "critical"),
        ("🏆 Top Revenue Category",
         f"**{top_rev_cat['category'][0]}** leads revenue generation. "
         f"Prioritize inventory and promotions in this segment.",
         "positive"),
        ("😡 Highest Complaint Category",
         f"**{neg_cat['category'][0]}** has the highest negative sentiment rate "
         f"({neg_cat['neg_pct'][0]}%). Product team should investigate.",
         "warning"),
        ("🛒 Cart Abandonment Issue",
         f"Only **{avg_purchase_ev['c'][0]:,} of {avg_abandon['c'][0]:,} cart events** "
         f"converted to purchase — abandonment rate of "
         f"**{100 - avg_purchase_ev['c'][0]/avg_abandon['c'][0]*100:.1f}%**. "
         f"Consider checkout flow optimization.",
         "warning"),
        ("🚨 Critical Products",
         f"**{critical_products['n'][0]} products** have a health score below 30. "
         f"These need urgent product quality or marketing intervention.",
         "critical"),
        ("👑 VIP Segment Revenue",
         f"**{top_seg_rev['segment'][0]} customers** contribute the most revenue "
         f"(${top_seg_rev['revenue'][0]:,.0f}). "
         f"Loyalty rewards will maximize their LTV.",
         "positive"),
        ("❌ Cancellation Rate",
         f"Order cancellation rate is **{cancel_rate['r'][0]}%**. "
         f"Benchmark suggests anything above 15% needs process review.",
         "warning"),
    ]

    for title, body, severity in insights:
        if severity == "critical":
            st.error(f"**{title}**\n\n{body}")
        elif severity == "warning":
            st.warning(f"**{title}**\n\n{body}")
        else:
            st.success(f"**{title}**\n\n{body}")

    st.markdown("---")
    st.subheader("📋 Recommended Actions")
    actions = [
        ("Retention Campaign",  "High", "Send personalized discount coupons to 70%+ churn risk customers within 7 days."),
        ("Product Quality Fix", "High", f"Investigate {neg_cat['category'][0]} category — highest complaint rate in reviews."),
        ("Checkout Optimization","Medium","A/B test simplified checkout to reduce cart abandonment."),
        ("VIP Loyalty Program",  "Medium","Launch exclusive rewards for VIP segment to maintain revenue concentration."),
        ("Health Score Monitor", "Low",  f"Set weekly alerts for the {critical_products['n'][0]} critical health score products."),
        ("Cancellation Survey",  "Low",  "Trigger exit survey on cancelled orders to identify root causes."),
    ]
    action_df = pd.DataFrame(actions, columns=["Action","Priority","Description"])
    st.dataframe(action_df, use_container_width=True, height=280)


# ── FOOTER ────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Customer Intelligence Platform v3 · 10K Users · 20K Orders · 15K Reviews · 80K Events · Python · Scikit-Learn · SQLite · Streamlit · Plotly")
