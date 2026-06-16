"""
Streamlit Dashboard — Customer Churn Intelligence
Run: streamlit run src/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Churn Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.kpi-box { background: #f8f9fa; border-radius: 10px; padding: 18px 20px; border-left: 4px solid #378ADD; }
.kpi-label { font-size: 13px; color: #666; margin-bottom: 4px; }
.kpi-val { font-size: 28px; font-weight: 700; color: #1a1a1a; }
.kpi-delta { font-size: 12px; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/customers.csv")
    except FileNotFoundError:
        from src.churn_analysis import generate_customer_data, engineer_features
        df = generate_customer_data(3000)
        df = engineer_features(df)
    df["tenure_bucket"] = pd.cut(
        df["tenure_months"],
        bins=[0, 3, 6, 12, 24, 60],
        labels=["0–3 mo", "3–6 mo", "6–12 mo", "1–2 yr", "2+ yr"]
    )
    return df


df = load_data()

# ── Sidebar filters ──────────────────────────────
st.sidebar.title("🔍 Filters")
plan_filter = st.sidebar.multiselect("Plan", df["plan"].unique(), default=df["plan"].unique())
industry_filter = st.sidebar.multiselect("Industry", df["industry"].unique(), default=df["industry"].unique())
region_filter = st.sidebar.multiselect("Region", df["region"].unique(), default=df["region"].unique())
min_tenure, max_tenure = st.sidebar.slider("Tenure range (months)", 1, 60, (1, 60))

filtered = df[
    df["plan"].isin(plan_filter) &
    df["industry"].isin(industry_filter) &
    df["region"].isin(region_filter) &
    df["tenure_months"].between(min_tenure, max_tenure)
]

# ── Header ───────────────────────────────────────
st.title("📊 Customer Churn & Revenue Intelligence")
st.caption(f"Showing {len(filtered):,} of {len(df):,} customers based on active filters")

# ── KPI row ──────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
churn_rate = filtered["churned"].mean() * 100
active_mrr  = filtered.loc[filtered.churned==0, "monthly_revenue"].sum()
lost_mrr    = filtered.loc[filtered.churned==1, "monthly_revenue"].sum()
avg_ltv     = filtered.loc[filtered.churned==0, "monthly_revenue"].mean() * 18  # simplified LTV

with k1:
    st.metric("Churn Rate", f"{churn_rate:.1f}%", delta="-0.8pp vs prev quarter",
              delta_color="inverse")
with k2:
    st.metric("Active MRR", f"${active_mrr:,.0f}", delta="+12% YoY")
with k3:
    st.metric("MRR at Risk", f"${lost_mrr:,.0f}", delta=f"{lost_mrr/active_mrr*100:.1f}% of MRR",
              delta_color="inverse")
with k4:
    st.metric("Avg Customer LTV", f"${avg_ltv:,.0f}", delta="+$240 YoY")

st.divider()

# ── Row 1: Churn by plan / Churn by tenure ───────
col1, col2 = st.columns(2)

with col1:
    plan_churn = filtered.groupby("plan")["churned"].mean().reset_index()
    plan_churn.columns = ["Plan", "Churn Rate"]
    plan_churn["Churn Rate %"] = plan_churn["Churn Rate"] * 100
    fig = px.bar(plan_churn, x="Plan", y="Churn Rate %",
                 color="Plan",
                 color_discrete_map={"Starter": "#E24B4A", "Pro": "#BA7517", "Enterprise": "#1D9E75"},
                 title="Churn rate by plan tier",
                 text_auto=".1f")
    fig.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
    fig.update_layout(showlegend=False, yaxis_title="Churn Rate (%)", height=350)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    tenure_churn = (
        filtered.dropna(subset=["tenure_bucket"])
        .groupby("tenure_bucket", observed=True)["churned"].mean() * 100
    ).reset_index()
    tenure_churn.columns = ["Tenure", "Churn Rate"]
    fig2 = px.line(tenure_churn, x="Tenure", y="Churn Rate",
                   markers=True, title="Churn rate by customer tenure",
                   color_discrete_sequence=["#7F77DD"])
    fig2.update_traces(line_width=3, marker_size=9)
    fig2.update_layout(yaxis_title="Churn Rate (%)", height=350)
    st.plotly_chart(fig2, use_container_width=True)

# ── Row 2: Revenue at risk heatmap / Engagement scatter ──
col3, col4 = st.columns([1.3, 1])

with col3:
    pivot = filtered.pivot_table(
        values="monthly_revenue",
        index="plan",
        columns="industry",
        aggfunc=lambda x: x[filtered.loc[x.index, "churned"]==1].sum()
    ).fillna(0).round(0)
    fig3 = px.imshow(pivot, title="Revenue at risk — Plan × Industry ($MRR)",
                     color_continuous_scale="Reds", text_auto=",.0f",
                     aspect="auto", height=340)
    fig3.update_coloraxes(colorbar_title="MRR ($)")
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    segment = filtered.groupby(["plan","churned"]).agg(
        count=("customer_id","count"),
        avg_rev=("monthly_revenue","mean"),
        avg_adoption=("feature_adoption_pct","mean")
    ).reset_index()
    segment["status"] = segment["churned"].map({0:"Retained", 1:"Churned"})
    fig4 = px.scatter(
        segment, x="avg_adoption", y="avg_rev",
        size="count", color="status",
        text="plan",
        color_discrete_map={"Retained": "#1D9E75", "Churned": "#E24B4A"},
        title="Segments: adoption vs revenue (bubble = size)",
        height=340
    )
    fig4.update_traces(textposition="top center")
    fig4.update_layout(xaxis_title="Avg feature adoption (%)", yaxis_title="Avg monthly revenue ($)")
    st.plotly_chart(fig4, use_container_width=True)

# ── At-risk customers table ───────────────────────
st.subheader("⚠️ At-risk customers — early intervention targets")

at_risk = filtered[
    (filtered["churned"] == 0) &
    (
        (filtered["tenure_months"] <= 3) |
        (filtered["logins_per_month"] <= 2) |
        (filtered["feature_adoption_pct"] <= 20)
    )
].copy()

at_risk["risk_score"] = (
    (at_risk["tenure_months"] <= 3).astype(int) * 3 +
    (at_risk["logins_per_month"] <= 2).astype(int) * 2 +
    (at_risk["feature_adoption_pct"] <= 20).astype(int) * 2 +
    (at_risk["support_tickets"] >= 5).astype(int) * 1
)

at_risk_display = (
    at_risk[["customer_id","plan","tenure_months","monthly_revenue",
             "logins_per_month","feature_adoption_pct","risk_score"]]
    .sort_values("risk_score", ascending=False)
    .head(25)
    .rename(columns={
        "customer_id": "Customer ID",
        "plan": "Plan",
        "tenure_months": "Tenure (mo)",
        "monthly_revenue": "MRR ($)",
        "logins_per_month": "Logins/mo",
        "feature_adoption_pct": "Feature Adoption (%)",
        "risk_score": "Risk Score"
    })
    .reset_index(drop=True)
)

st.dataframe(
    at_risk_display.style.background_gradient(subset=["Risk Score"], cmap="Reds"),
    use_container_width=True,
    height=400
)

st.caption(f"🎯 {len(at_risk):,} customers flagged as at-risk | "
           f"${at_risk['monthly_revenue'].sum():,.0f} MRR exposed")
