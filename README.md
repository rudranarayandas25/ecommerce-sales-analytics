# E-Commerce Sales Analytics

Comprehensive business intelligence dashboard for e-commerce operations. Data analytics capstone project.

## Project Overview

- **Dataset:** 25,000 orders from 2,000 customers (2024-2025), 5 product categories, 15 Indian cities
- **Analytics:** Revenue trends, RFM segmentation, cohort retention, product performance, geographic insights
- **Dashboard:** 6 interactive tabs built with Streamlit and Plotly

## Quick Start

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn plotly streamlit

# Generate data (already done)
python3 src/generate_data.py

# Launch dashboard
streamlit run app/dashboard.py

# Explore notebook
jupyter notebook notebooks/sales_analytics.ipynb
```

## Dashboard Tabs

| Tab | Content |
|---|---|
| Executive Summary | KPIs, revenue trends, order status, payment methods |
| Revenue & Trends | Time series, weekday/weekend analysis |
| Product Analytics | Category breakdown, top products, discount impact |
| Customer Analytics | RFM stats, frequency distribution, cohort retention |
| Geographic Insights | City map, regional revenue, city rankings |
| RFM Segmentation | 7 customer segments with recommendations |
