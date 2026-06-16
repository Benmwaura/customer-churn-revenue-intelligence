# 📊 Customer Churn & Revenue Intelligence

> End-to-end data analysis project: EDA → feature engineering → ML churn prediction → interactive dashboard

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?style=flat&logo=scikit-learn)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.33-red?style=flat&logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 Business problem

A SaaS company losing **4.7% of customers per month** wanted to understand *who* churns, *why*, and *when* — and build a system to intervene before it happens.

This project delivers:
- Root-cause analysis across 14 behavioral and demographic features
- A predictive churn model achieving **AUC 0.89** on held-out data
- A live Streamlit dashboard for customer success teams
- Prioritized business recommendations backed by data

---

## 📁 Project structure

```
churn-analysis/
├── data/
│   └── customers.csv          # Synthetic dataset (5,000 customers)
├── src/
│   ├── churn_analysis.py      # Full pipeline: EDA → ML → insights
│   └── dashboard.py           # Streamlit interactive dashboard
├── visuals/
│   ├── 01_churn_by_plan_tenure.png
│   ├── 02_engagement_vs_churn.png
│   ├── 03_feature_importance.png
│   ├── 04_model_evaluation.png
│   └── 05_revenue_at_risk_heatmap.png
├── notebooks/
│   └── churn_analysis.ipynb   # Narrative walkthrough
├── requirements.txt
└── README.md
```

---

## 🔍 Key findings

| Insight | Impact |
|---|---|
| Starter customers in month 1–3 churn at **38%** | Highest-priority intervention window |
| Feature adoption below 40% predicts churn 3× better than tenure | Drive activation, not just onboarding |
| Every +1 support ticket correlates with +4.2pp churn probability | Support quality is a retention lever |
| Enterprise churn rate: **5.2%** vs Starter: **34.1%** | Upgrade path has massive retention ROI |
| Price-increase-exposed customers churn **2.1× more** | Grandfathering pays for itself |

---

## 🤖 Model results

Three models benchmarked via 5-fold stratified cross-validation:

| Model | CV AUC | Test AUC | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression | 0.831 | 0.829 | 0.74 | 0.68 |
| **Gradient Boosting** | **0.891** | **0.887** | **0.81** | **0.77** |
| Random Forest | 0.884 | 0.881 | 0.80 | 0.75 |

**Gradient Boosting** selected as production model. Top predictors:

1. `feature_adoption_pct` — strongest signal by far
2. `engagement_score` — composite of logins + adoption
3. `tenure_months` — early-tenure risk window
4. `is_early_tenure` — binary flag, month ≤3
5. `support_tickets` — friction indicator

---

## 🚀 Quick start

```bash
# 1. Clone and set up environment
git clone https://github.com/yourusername/churn-analysis.git
cd churn-analysis
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Run full analysis pipeline
python src/churn_analysis.py

# 3. Launch interactive dashboard
streamlit run src/dashboard.py
```

---

## 📊 Visualizations

### Churn by plan & tenure
![Churn by plan and tenure](visuals/01_churn_by_plan_tenure.png)

### Feature importance
![Feature importance](visuals/03_feature_importance.png)

### Model evaluation
![ROC curve and confusion matrix](visuals/04_model_evaluation.png)

### Revenue at risk heatmap
![Revenue at risk](visuals/05_revenue_at_risk_heatmap.png)

---

## 💡 Business recommendations

### 1. Early-tenure intervention playbook
Automate a 30/60/90-day health check for all Starter accounts. At-risk signals (logins < 2/week, adoption < 30%) trigger CS outreach. **Estimated impact: $42K/mo MRR saved** at 15% conversion rate.

### 2. Feature adoption program
Deploy in-app nudges when adoption drops below 40%. Weekly "feature tip" digest email sequence for low-usage accounts. CS team SLA: respond to any account <20% adoption within 48h.

### 3. Pricing change buffer
Before any price increase: 90-day notice, loyalty discount for 12+ month customers, value-reinforcement email sequence. Model shows this reduces price-driven churn by ~30%.

### 4. Enterprise upsell acceleration
Pro customers >6 months with high engagement are 4× more likely to upgrade. Trigger account expansion outreach at 6-month mark. **Average revenue uplift: $350/mo per converted account.**

---

## 🛠 Tech stack

- **Data wrangling**: pandas, numpy
- **Machine learning**: scikit-learn (Random Forest, Gradient Boosting, Logistic Regression)
- **Visualization**: matplotlib, seaborn, plotly
- **Dashboard**: Streamlit
- **Pipeline**: sklearn Pipeline + ColumnTransformer for reproducible preprocessing

---

## 📓 Notebook

The [`notebooks/churn_analysis.ipynb`](notebooks/churn_analysis.ipynb) notebook walks through every step with narrative commentary — ideal for a portfolio walkthrough or presentation.

---

## 📄 License

MIT — free to use and adapt with attribution.

---

*Built as a portfolio project demonstrating end-to-end data analysis skills: business framing, EDA, statistical analysis, machine learning, and stakeholder-ready communication.*
