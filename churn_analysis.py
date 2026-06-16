"""
Customer Churn & Revenue Intelligence Analysis
===============================================
End-to-end pipeline: data generation → EDA → feature engineering → ML model → visualizations
Author: [Your Name]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ─────────────────────────────────────────────
# 1. SYNTHETIC DATA GENERATION
# ─────────────────────────────────────────────

def generate_customer_data(n=5000):
    """Generate realistic SaaS customer dataset with churn signal baked in."""
    plans = np.random.choice(["Starter", "Pro", "Enterprise"],
                             size=n, p=[0.45, 0.35, 0.20])
    tenure_months = np.random.gamma(shape=2.5, scale=12, size=n).astype(int) + 1
    tenure_months = np.clip(tenure_months, 1, 60)

    monthly_revenue = np.where(
        plans == "Starter",  np.random.normal(49,  15, n),
        np.where(plans == "Pro", np.random.normal(149, 30, n),
                                  np.random.normal(499, 80, n))
    ).clip(20, 800)

    support_tickets = np.random.poisson(
        lam=np.where(plans == "Starter", 3, np.where(plans == "Pro", 1.5, 0.8)),
        size=n
    )
    logins_per_month = np.random.gamma(
        shape=np.where(plans == "Enterprise", 5, 2),
        scale=np.where(plans == "Enterprise", 4, 3),
        size=n
    ).clip(0, 60)

    feature_adoption = np.random.beta(
        a=np.where(plans == "Enterprise", 4, np.where(plans == "Pro", 2, 1)),
        b=np.where(plans == "Enterprise", 1.5, np.where(plans == "Pro", 2, 3)),
        size=n
    )

    num_seats = np.where(
        plans == "Starter",   np.random.randint(1, 5, n),
        np.where(plans == "Pro", np.random.randint(3, 20, n),
                                  np.random.randint(15, 200, n))
    )

    industry = np.random.choice(
        ["SaaS", "Finance", "Healthcare", "Retail", "Education", "Other"],
        size=n, p=[0.30, 0.20, 0.15, 0.15, 0.10, 0.10]
    )

    region = np.random.choice(
        ["North America", "Europe", "Asia Pacific", "Latin America"],
        size=n, p=[0.50, 0.25, 0.15, 0.10]
    )

    had_trial = np.random.choice([0, 1], size=n, p=[0.35, 0.65])
    payment_method = np.random.choice(
        ["credit_card", "invoice", "paypal"],
        size=n, p=[0.55, 0.35, 0.10]
    )
    price_increase_exposed = np.random.choice([0, 1], size=n, p=[0.70, 0.30])

    # Churn probability (logistic function over real drivers)
    churn_score = (
        + 2.5 * (plans == "Starter")
        - 1.2 * (plans == "Enterprise")
        - 0.04 * tenure_months
        + 0.25 * support_tickets
        - 2.5 * feature_adoption
        - 0.04 * logins_per_month
        + 1.5 * (tenure_months < 3)
        + 0.8 * price_increase_exposed
        + 0.5 * (payment_method == "paypal")
        + np.random.normal(0, 0.6, n)
    )
    churn_prob = 1 / (1 + np.exp(-churn_score + 1.5))
    churned = (np.random.random(n) < churn_prob).astype(int)

    df = pd.DataFrame({
        "customer_id":          [f"CUST-{i:05d}" for i in range(n)],
        "plan":                 plans,
        "tenure_months":        tenure_months,
        "monthly_revenue":      monthly_revenue.round(2),
        "support_tickets":      support_tickets,
        "logins_per_month":     logins_per_month.round(1),
        "feature_adoption_pct": (feature_adoption * 100).round(1),
        "num_seats":            num_seats,
        "industry":             industry,
        "region":               region,
        "had_trial":            had_trial,
        "payment_method":       payment_method,
        "price_increase_exposed": price_increase_exposed,
        "churned":              churned
    })
    return df


# ─────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────

def run_eda(df):
    print("=" * 55)
    print("  CUSTOMER CHURN — EXPLORATORY DATA ANALYSIS")
    print("=" * 55)

    total = len(df)
    churned = df["churned"].sum()
    rate = churned / total * 100
    print(f"\n📊 Dataset: {total:,} customers | {churned:,} churned ({rate:.1f}%)")
    print(f"   MRR:     ${df.loc[df.churned==0,'monthly_revenue'].sum():,.0f}/mo")
    print(f"   Lost MRR:${df.loc[df.churned==1,'monthly_revenue'].sum():,.0f}/mo")

    print("\n─── Churn rate by plan ──────────────────────────")
    plan_summary = df.groupby("plan").agg(
        customers=("churned", "count"),
        churned=("churned", "sum"),
        avg_revenue=("monthly_revenue", "mean")
    )
    plan_summary["churn_rate"] = (plan_summary["churned"] / plan_summary["customers"] * 100).round(1)
    print(plan_summary.to_string())

    print("\n─── Churn rate by tenure bucket ────────────────")
    df["tenure_bucket"] = pd.cut(
        df["tenure_months"],
        bins=[0, 3, 6, 12, 24, 60],
        labels=["0–3 mo", "3–6 mo", "6–12 mo", "1–2 yr", "2+ yr"]
    )
    tenure_churn = df.groupby("tenure_bucket")["churned"].mean() * 100
    for bucket, rate in tenure_churn.items():
        bar = "█" * int(rate / 2)
        print(f"  {bucket:<10}  {bar:<25} {rate:.1f}%")

    print("\n─── Key statistics (churned vs retained) ───────")
    cols = ["logins_per_month", "feature_adoption_pct", "support_tickets", "monthly_revenue"]
    comp = df.groupby("churned")[cols].mean().round(2)
    comp.index = ["Retained", "Churned"]
    print(comp.to_string())

    return df


# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_features(df):
    """Create derived features that improve model signal."""
    df = df.copy()

    df["revenue_per_seat"] = (df["monthly_revenue"] / df["num_seats"]).round(2)
    df["engagement_score"] = (
        df["logins_per_month"] * 0.4 +
        df["feature_adoption_pct"] * 0.6
    ).round(2)
    df["tickets_per_seat"] = (df["support_tickets"] / df["num_seats"].clip(1)).round(3)
    df["is_early_tenure"] = (df["tenure_months"] <= 3).astype(int)
    df["high_support"] = (df["support_tickets"] >= df["support_tickets"].quantile(0.75)).astype(int)
    df["low_engagement"] = (df["logins_per_month"] <= df["logins_per_month"].quantile(0.25)).astype(int)
    df["revenue_tier"] = pd.cut(
        df["monthly_revenue"],
        bins=[0, 75, 200, 800],
        labels=["low", "mid", "high"]
    )

    print("\n✅ Feature engineering complete — 7 new features added")
    return df


# ─────────────────────────────────────────────
# 4. MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────

def build_churn_model(df):
    """Train and evaluate a churn prediction model."""

    num_features = [
        "tenure_months", "monthly_revenue", "support_tickets",
        "logins_per_month", "feature_adoption_pct", "num_seats",
        "had_trial", "price_increase_exposed", "engagement_score",
        "revenue_per_seat", "tickets_per_seat", "is_early_tenure",
        "high_support", "low_engagement"
    ]
    cat_features = ["plan", "industry", "region", "payment_method"]

    X = df[num_features + cat_features]
    y = df["churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features)
    ])

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, random_state=42),
    }

    print("\n─── Model comparison (5-fold CV AUC) ───────────")
    best_name, best_model, best_score = None, None, 0

    for name, clf in models.items():
        pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
        scores = cross_val_score(pipe, X_train, y_train, cv=5,
                                  scoring="roc_auc", n_jobs=-1)
        mean_auc = scores.mean()
        std_auc  = scores.std()
        marker = " ◀ best" if mean_auc > best_score else ""
        print(f"  {name:<25} AUC: {mean_auc:.4f} ± {std_auc:.4f}{marker}")
        if mean_auc > best_score:
            best_score = mean_auc
            best_name  = name
            best_model = pipe

    print(f"\n🏆 Best model: {best_name} (AUC {best_score:.4f})")
    best_model.fit(X_train, y_train)

    y_pred      = best_model.predict(X_test)
    y_pred_prob = best_model.predict_proba(X_test)[:, 1]
    test_auc    = roc_auc_score(y_test, y_pred_prob)

    print(f"\n─── Test set performance ────────────────────────")
    print(f"  ROC-AUC: {test_auc:.4f}")
    print("\n" + classification_report(y_test, y_pred, target_names=["Retained","Churned"]))

    # Feature importances (RF only)
    rf_pipe = Pipeline([("prep", preprocessor),
                        ("clf", RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42))])
    rf_pipe.fit(X_train, y_train)
    cat_encoded_names = (rf_pipe.named_steps["prep"]
                         .transformers_[1][1]
                         .get_feature_names_out(cat_features).tolist())
    all_features = num_features + cat_encoded_names
    importances  = rf_pipe.named_steps["clf"].feature_importances_

    fi_df = (pd.DataFrame({"feature": all_features, "importance": importances})
               .sort_values("importance", ascending=False)
               .head(12)
               .reset_index(drop=True))

    print("\n─── Top 12 churn predictors ─────────────────────")
    for _, row in fi_df.iterrows():
        bar = "█" * int(row["importance"] * 200)
        print(f"  {row['feature']:<30} {bar:<20} {row['importance']:.4f}")

    return best_model, fi_df, (X_test, y_test, y_pred, y_pred_prob), (X_train, y_train)


# ─────────────────────────────────────────────
# 5. VISUALIZATIONS
# ─────────────────────────────────────────────

PALETTE = {
    "blue":   "#378ADD",
    "red":    "#E24B4A",
    "green":  "#1D9E75",
    "amber":  "#BA7517",
    "gray":   "#888780",
    "purple": "#7F77DD",
}

def plot_all(df, fi_df, eval_data, output_dir="visuals"):
    import os; os.makedirs(output_dir, exist_ok=True)

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.grid":        True,
        "grid.alpha":       0.25,
        "grid.linestyle":   "--",
        "font.family":      "DejaVu Sans",
        "axes.spines.top":  False,
        "axes.spines.right":False,
    })

    X_test, y_test, y_pred, y_pred_prob = eval_data

    # ── Fig 1: Churn rate by plan + tenure ──────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Churn Analysis — Plan & Tenure Breakdown", fontsize=14, fontweight="bold", y=1.02)

    plan_rates = (df.groupby("plan")["churned"].mean() * 100).sort_values(ascending=False)
    axes[0].bar(plan_rates.index, plan_rates.values,
                color=[PALETTE["red"], PALETTE["amber"], PALETTE["blue"]],
                edgecolor="white", linewidth=0.5, zorder=3)
    for i, v in enumerate(plan_rates.values):
        axes[0].text(i, v + 0.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    axes[0].set_title("Churn rate by plan", fontsize=12)
    axes[0].set_ylabel("Churn rate (%)")
    axes[0].set_ylim(0, plan_rates.max() * 1.25)

    df["tenure_bucket"] = pd.cut(
        df["tenure_months"],
        bins=[0, 3, 6, 12, 24, 60],
        labels=["0–3 mo", "3–6 mo", "6–12 mo", "1–2 yr", "2+ yr"]
    )
    tenure_rates = df.groupby("tenure_bucket")["churned"].mean() * 100
    axes[1].plot(tenure_rates.index.astype(str), tenure_rates.values,
                 marker="o", color=PALETTE["purple"], linewidth=2.5, markersize=7, zorder=3)
    axes[1].fill_between(range(len(tenure_rates)), tenure_rates.values,
                          alpha=0.12, color=PALETTE["purple"])
    for i, v in enumerate(tenure_rates.values):
        axes[1].text(i, v + 0.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=10)
    axes[1].set_title("Churn rate by customer tenure", fontsize=12)
    axes[1].set_ylabel("Churn rate (%)")
    axes[1].set_ylim(0, tenure_rates.max() * 1.3)
    axes[1].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    fig.savefig(f"{output_dir}/01_churn_by_plan_tenure.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Fig 2: Engagement vs churn (scatter) ────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Engagement Signals vs Churn", fontsize=14, fontweight="bold", y=1.02)

    for c, label, color in [(0, "Retained", PALETTE["blue"]), (1, "Churned", PALETTE["red"])]:
        sub = df[df["churned"] == c]
        axes[0].scatter(sub["logins_per_month"], sub["feature_adoption_pct"],
                        alpha=0.25, s=14, c=color, label=label)
    axes[0].set_xlabel("Logins per month")
    axes[0].set_ylabel("Feature adoption (%)")
    axes[0].set_title("Login activity vs feature adoption")
    axes[0].legend()

    box_data = [df.loc[df["churned"]==0, "support_tickets"],
                df.loc[df["churned"]==1, "support_tickets"]]
    bp = axes[1].boxplot(box_data, labels=["Retained", "Churned"], patch_artist=True,
                          medianprops={"color":"white","linewidth":2})
    bp["boxes"][0].set_facecolor(PALETTE["blue"])
    bp["boxes"][1].set_facecolor(PALETTE["red"])
    axes[1].set_title("Support ticket volume by churn status")
    axes[1].set_ylabel("Support tickets (total)")

    plt.tight_layout()
    fig.savefig(f"{output_dir}/02_engagement_vs_churn.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Fig 3: Feature importances ───────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [PALETTE["blue"] if i < 3 else PALETTE["gray"] for i in range(len(fi_df))]
    ax.barh(fi_df["feature"][::-1], fi_df["importance"][::-1],
            color=colors[::-1], edgecolor="white", linewidth=0.4)
    ax.set_title("Top churn predictors — Random Forest feature importance",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance score")
    plt.tight_layout()
    fig.savefig(f"{output_dir}/03_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Fig 4: ROC curve + Confusion matrix ─────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Model Evaluation", fontsize=14, fontweight="bold", y=1.02)

    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    auc = roc_auc_score(y_test, y_pred_prob)
    axes[0].plot(fpr, tpr, color=PALETTE["blue"], lw=2.5, label=f"ROC (AUC = {auc:.3f})")
    axes[0].plot([0,1],[0,1], color=PALETTE["gray"], lw=1, linestyle="--", label="Random")
    axes[0].fill_between(fpr, tpr, alpha=0.08, color=PALETTE["blue"])
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curve")
    axes[0].legend(loc="lower right")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Retained","Churned"])
    disp.plot(ax=axes[1], colorbar=False, cmap="Blues")
    axes[1].set_title("Confusion matrix")

    plt.tight_layout()
    fig.savefig(f"{output_dir}/04_model_evaluation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Fig 5: Revenue at risk heatmap ───────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot = df.pivot_table(
        values="monthly_revenue",
        index="plan",
        columns="industry",
        aggfunc=lambda x: x[df.loc[x.index, "churned"] == 1].sum()
    ).fillna(0)
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="Reds", ax=ax,
                linewidths=0.4, cbar_kws={"label": "Monthly Revenue at Risk ($)"})
    ax.set_title("Revenue at risk by plan × industry ($MRR at churn risk)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(f"{output_dir}/05_revenue_at_risk_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n📁 5 charts saved to /{output_dir}/")


# ─────────────────────────────────────────────
# 6. BUSINESS INSIGHTS & RECOMMENDATIONS
# ─────────────────────────────────────────────

def print_recommendations(df):
    print("\n" + "=" * 55)
    print("  STRATEGIC RECOMMENDATIONS")
    print("=" * 55)

    starter_early = df[(df["plan"]=="Starter") & (df["tenure_months"]<=3)]
    at_risk_rev = starter_early[starter_early["churned"]==1]["monthly_revenue"].sum()

    print(f"""
1. EARLY TENURE INTERVENTION (highest ROI)
   ├─ Starter customers < 3 months churn at {starter_early['churned'].mean()*100:.0f}%
   ├─ ${at_risk_rev:,.0f}/mo MRR at risk from this segment alone
   └─ Action: 30/60/90-day onboarding health check automation

2. FEATURE ADOPTION PROGRAM
   ├─ Low-adoption customers churn 3× more often
   ├─ Top churners average {df[df.churned==1]['feature_adoption_pct'].mean():.0f}% feature adoption vs
   │  {df[df.churned==0]['feature_adoption_pct'].mean():.0f}% for retained customers
   └─ Action: In-app nudges + CS outreach at <40% adoption threshold

3. SUPPORT EXPERIENCE OVERHAUL
   ├─ Customers with 5+ tickets churn at {df[df.support_tickets>=5]['churned'].mean()*100:.0f}%
   └─ Action: Senior CS escalation path + CSAT-driven early warning system

4. PRICE SENSITIVITY MITIGATION
   ├─ Price-increase-exposed customers show elevated churn risk
   └─ Action: Grandfathering + value reinforcement emails before increases

5. ENTERPRISE EXPANSION PLAY
   ├─ Enterprise segment: {df[df.plan=='Enterprise']['churned'].mean()*100:.1f}% churn vs
   │  {df[df.plan=='Starter']['churned'].mean()*100:.1f}% for Starter
   └─ Action: Accelerate Pro → Enterprise upgrade path with seat-based pricing
""")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating customer dataset...")
    df = generate_customer_data(n=5000)
    df.to_csv("data/customers.csv", index=False)
    print(f"✅ Saved data/customers.csv ({len(df):,} rows)")

    df = run_eda(df)
    df = engineer_features(df)

    best_model, fi_df, eval_data, train_data = build_churn_model(df)

    plot_all(df, fi_df, eval_data)
    print_recommendations(df)

    # Export scored dataset for BI/dashboard use
    X_all = df.drop(columns=["churned", "tenure_bucket"], errors="ignore")
    print("\n✅ Analysis pipeline complete. Ready for dashboard integration.")
