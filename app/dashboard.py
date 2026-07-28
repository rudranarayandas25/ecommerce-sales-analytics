import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

st.set_page_config(
    page_title="E-Commerce Sales Analytics",
    page_icon="shopping_cart",
    layout="wide",
)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_data
def load_data():
    orders = pd.read_csv(os.path.join(BASE, "data", "orders.csv"), parse_dates=["order_date"])
    customers = pd.read_csv(os.path.join(BASE, "data", "customers.csv"))
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    df = orders.merge(customers, on="customer_id", how="left")
    return df, orders, customers

df, orders, customers = load_data()

st.title("E-Commerce Sales Analytics Dashboard")
st.markdown("Comprehensive business intelligence and analytics for e-commerce operations (2024-2025)")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Executive Summary", "Revenue & Trends", "Product Analytics",
    "Customer Analytics", "Geographic Insights", "RFM Segmentation"
])

@st.cache_data
def compute_rfm(_orders, _customers):
    ref_date = _orders["order_date"].max() + pd.Timedelta(days=1)
    rfm_df = _orders.groupby("customer_id").agg(
        Recency=("order_date", lambda x: (ref_date - x.max()).days),
        Frequency=("order_id", "count"),
        Monetary=("total_amount", "sum"),
    ).reset_index()
    return rfm_df.merge(_customers[["customer_id", "city", "region"]], on="customer_id", how="left")

@st.cache_data
def compute_cohort(_orders):
    o = _orders.copy()
    o["cohort_month"] = o.groupby("customer_id")["order_date"].transform("min").dt.to_period("M")
    o["order_month"] = o["order_date"].dt.to_period("M")
    o["cohort_index"] = (o["order_month"] - o["cohort_month"]).apply(lambda x: x.n if hasattr(x, "n") else 0)
    cohort_data = o.groupby(["cohort_month", "cohort_index"]).agg(customers=("customer_id", "nunique")).reset_index()
    cohort_pivot = cohort_data.pivot_table(index="cohort_month", columns="cohort_index", values="customers", aggfunc="sum")
    mx = min(12, len(cohort_pivot.columns))
    cohort_pivot = cohort_pivot.iloc[:, :mx]
    return (cohort_pivot.divide(cohort_pivot.iloc[:, 0], axis=0) * 100).round(1)

@st.cache_data
def compute_segments(_orders, _customers):
    ref_date = _orders["order_date"].max() + pd.Timedelta(days=1)
    r = _orders.groupby("customer_id").agg(
        Recency=("order_date", lambda x: (ref_date - x.max()).days),
        Frequency=("order_id", "count"),
        Monetary=("total_amount", "sum"),
    ).reset_index()
    r = r.merge(_customers[["customer_id", "city", "region"]], on="customer_id", how="left")
    r["R_Score"] = pd.qcut(r["Recency"].rank(method="first"), 4, labels=[4, 3, 2, 1])
    r["F_Score"] = pd.qcut(r["Frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4])
    r["M_Score"] = pd.qcut(r["Monetary"].rank(method="first"), 4, labels=[1, 2, 3, 4])
    def seg(row):
        rr, ff, mm = int(row["R_Score"]), int(row["F_Score"]), int(row["M_Score"])
        if rr >= 4 and ff >= 4 and mm >= 4: return "Champions"
        elif rr >= 3 and ff >= 3 and mm >= 3: return "Loyal Customers"
        elif rr >= 4 and ff <= 2: return "New Customers"
        elif rr <= 2 and ff >= 3 and mm >= 3: return "At Risk"
        elif rr <= 2 and ff <= 2 and mm <= 2: return "Lost"
        elif ff >= 3: return "Potential Loyalists"
        return "Need Attention"
    r["Segment"] = r.apply(seg, axis=1)
    return r

rfm_all = compute_rfm(orders, customers)
rfm_seg = compute_segments(orders, customers)
cohort_pct = compute_cohort(orders)

seg_colors = {
    "Champions": "#006837", "Loyal Customers": "#1a9850",
    "Potential Loyalists": "#66bd63", "New Customers": "#a6d96a",
    "At Risk": "#fdae61", "Need Attention": "#f46d43", "Lost": "#d73027",
}

# ============================================================
# TAB 1: Executive Summary
# ============================================================
with tab1:
    st.header("Executive Summary")
    tr = orders["total_amount"].sum()
    to = len(orders)
    tc = len(customers)
    av = orders["total_amount"].mean()
    de = len(orders[orders["status"] == "Delivered"])
    dr = de / to * 100

    cols = st.columns(6)
    cols[0].metric("Total Revenue", f"INR {tr/1e7:.1f} Cr")
    cols[1].metric("Total Orders", f"{to:,}")
    cols[2].metric("Customers", f"{tc:,}")
    cols[3].metric("Avg Order Value", f"INR {av:,.0f}")
    cols[4].metric("Delivery Rate", f"{dr:.1f}%")
    cols[5].metric("Categories", f"{orders['category'].nunique()}")

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        monthly = orders[orders["status"] == "Delivered"].copy()
        monthly["ym"] = monthly["order_date"].dt.to_period("M").astype(str)
        monthly = monthly.groupby("ym").agg(Revenue=("total_amount", "sum"), Orders=("order_id", "count")).reset_index()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=monthly["ym"], y=monthly["Revenue"] / 1e6, name="Revenue (M)", marker_color="#636EFA"), secondary_y=False)
        fig.add_trace(go.Scatter(x=monthly["ym"], y=monthly["Orders"], name="Orders", mode="lines+markers", line=dict(color="#EF553B", width=2)), secondary_y=True)
        fig.update_layout(title="Monthly Revenue & Order Volume", height=400, hovermode="x unified", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        cr = orders.groupby("category")["total_amount"].sum().sort_values()
        fig = px.bar(x=cr.values / 1e6, y=cr.index, orientation="h", title="Revenue by Category",
                     labels={"x": "Revenue (Millions INR)", "y": ""},
                     color=cr.values, color_continuous_scale="Blues",
                     text=(cr.values / 1e6).round(1))
        fig.update_traces(texttemplate="INR %{text}M", textposition="outside")
        fig.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        sc = orders["status"].value_counts()
        fig = px.pie(values=sc.values, names=sc.index, title="Order Status Distribution",
                     color_discrete_sequence=px.colors.qualitative.Set2, hole=0.4)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(height=350, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        pm = orders["payment_method"].value_counts()
        fig = px.pie(values=pm.values, names=pm.index, title="Payment Method Distribution",
                     color_discrete_sequence=px.colors.qualitative.Pastel, hole=0.4)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(height=350, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 2: Revenue & Trends
# ============================================================
with tab2:
    st.header("Revenue & Trend Analysis")
    vo = st.radio("View by:", ["Monthly", "Quarterly", "Weekly"], horizontal=True)
    fm = {"Monthly": "M", "Quarterly": "Q", "Weekly": "W"}
    tc2 = orders["order_date"].dt.to_period(fm[vo]).astype(str)
    td = orders.assign(period=tc2).groupby("period").agg(
        Revenue=("total_amount", "sum"), Orders=("order_id", "count"),
        AOV=("total_amount", "mean"), Customers=("customer_id", "nunique"),
    ).reset_index()

    mcols = st.columns(4)
    mcols[0].metric("Peak Revenue", f"INR {td['Revenue'].max()/1e6:.1f}M")
    mcols[1].metric("Peak Orders", f"{td['Orders'].max():,}")
    mcols[2].metric("Best AOV", f"INR {td['AOV'].max():,.0f}")
    mcols[3].metric("Active Customers (max)", f"{td['Customers'].max():,}")

    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "Revenue Over Time", "Order Count Over Time",
        "Average Order Value Trend", "Active Customers Trend",
    ), vertical_spacing=0.14)
    fig.add_trace(go.Scatter(x=td["period"], y=td["Revenue"] / 1e6, mode="lines+markers", fill="tozeroy",
                             line=dict(color="#636EFA", width=2)), row=1, col=1)
    fig.add_trace(go.Bar(x=td["period"], y=td["Orders"], marker_color="#00CC96"), row=1, col=2)
    fig.add_trace(go.Scatter(x=td["period"], y=td["AOV"], mode="lines+markers",
                             line=dict(color="#AB63FA", width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=td["period"], y=td["Customers"], mode="lines+markers", fill="tozeroy",
                             line=dict(color="#FFA15A", width=2)), row=2, col=2)
    fig.update_layout(height=600, showlegend=False, template="plotly_white",
                      title_text=f"{vo} Revenue & Order Trends")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Day of Week & Weekend Analysis")
    c1, c2 = st.columns(2)
    with c1:
        do = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dr2 = orders.groupby("day_of_week")["total_amount"].sum().reindex(do)
        fig = px.bar(x=dr2.index, y=dr2.values / 1e6, title="Revenue by Day of Week",
                     labels={"x": "", "y": "Revenue (Millions INR)"},
                     color=dr2.values, color_continuous_scale="Viridis")
        fig.update_layout(height=350, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        wk = orders.groupby("is_weekend")["total_amount"].sum()
        wkdf = pd.DataFrame({"Period": ["Weekday", "Weekend"], "Revenue": [wk.get(False, 0), wk.get(True, 0)]})
        fig = px.pie(wkdf, values="Revenue", names="Period", title="Revenue Split: Weekday vs Weekend",
                     color_discrete_sequence=["#636EFA", "#EF553B"], hole=0.5)
        fig.update_layout(height=350, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 3: Product Analytics
# ============================================================
with tab3:
    st.header("Product & Category Analytics")
    cs2 = orders.groupby("category").agg(
        Revenue=("total_amount", "sum"), Orders=("order_id", "count"),
        AOV=("total_amount", "mean"), Products=("product", "nunique"),
    ).sort_values("Revenue", ascending=False).reset_index()
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(cs2, x="Revenue", y="category", orientation="h", title="Category Revenue Breakdown",
                     color="Revenue", color_continuous_scale="Blues",
                     text=cs2["Revenue"].apply(lambda x: f"INR {x/1e6:.1f}M"))
        fig.update_traces(textposition="outside")
        fig.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.scatter(cs2, x="Orders", y="AOV", size="Revenue", color="category", text="category",
                         title="Category Performance Matrix",
                         labels={"Orders": "Total Orders", "AOV": "Avg Order Value (INR)"}, size_max=55)
        fig.update_traces(textposition="top center")
        fig.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 10 Products by Revenue")
    tp = orders.groupby(["category", "product"]).agg(
        Revenue=("total_amount", "sum"), Orders=("order_id", "count"), AOV=("total_amount", "mean"),
    ).sort_values("Revenue", ascending=False).head(10).reset_index()
    fig = px.bar(tp, x="Revenue", y="product", color="category", orientation="h",
                 title="Top 10 Products by Revenue",
                 text=tp["Revenue"].apply(lambda x: f"INR {x/1e6:.1f}M"),
                 labels={"Revenue": "Revenue (INR)", "product": ""},
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_traces(textposition="outside")
    fig.update_layout(height=400, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(tp.style.format({"Revenue": "INR {:,.0f}", "AOV": "INR {:,.0f}", "Orders": "{:,}"}), use_container_width=True)

    st.subheader("Discount Impact Analysis")
    d = orders.copy()
    d["db"] = pd.cut(d["discount_pct"], bins=[-1, 0, 10, 20, 100], labels=["No Discount", "1-10%", "11-20%", "21-25%"])
    da = d.groupby("db").agg(Orders=("order_id", "count"), Revenue=("total_amount", "sum"), AOV=("total_amount", "mean")).reset_index()
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(da, x="db", y="Orders", title="Orders by Discount Level", color="db",
                     color_discrete_sequence=px.colors.qualitative.Set2, text="Orders")
        fig.update_traces(textposition="outside")
        fig.update_layout(height=350, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(da, x="db", y="Revenue", title="Revenue by Discount Level", color="db",
                     color_discrete_sequence=px.colors.qualitative.Set2,
                     text=da["Revenue"].apply(lambda x: f"INR {x/1e6:.1f}M"))
        fig.update_traces(textposition="outside")
        fig.update_layout(height=350, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 4: Customer Analytics
# ============================================================
with tab4:
    st.header("Customer Analytics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Recency", f"{rfm_all['Recency'].mean():.0f} days")
    c2.metric("Avg Frequency", f"{rfm_all['Frequency'].mean():.1f} orders")
    c3.metric("Avg Monetary", f"INR {rfm_all['Monetary'].mean():,.0f}")

    st.subheader("Customer Order Frequency Distribution")
    fd = rfm_all["Frequency"].value_counts().sort_index().reset_index()
    fd.columns = ["Orders", "Customers"]
    fig = px.bar(fd, x="Orders", y="Customers", title="How Many Orders Do Customers Place?",
                 color="Customers", color_continuous_scale="Blues", text="Customers")
    fig.update_traces(textposition="outside")
    fig.update_layout(height=400, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        sn = min(500, len(rfm_all))
        fig = px.scatter(rfm_all.sample(sn), x="Frequency", y="Monetary", color="Recency",
                         title="Customer Value Scatter",
                         labels={"Frequency": "Order Count", "Monetary": "Total Spend (INR)", "Recency": "Days Since Last Order"},
                         color_continuous_scale="RdYlGn_r", opacity=0.7)
        fig.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(rfm_all, x="Recency", nbins=40, title="Days Since Last Order Distribution",
                           color_discrete_sequence=["#636EFA"], labels={"Recency": "Days Since Last Order"})
        fig.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Customer Retention (Cohort Analysis)")
    cd2 = cohort_pct.copy()
    cd2.index = cd2.index.astype(str)
    cd2.columns = [str(c) for c in cd2.columns]
    annot = cd2.copy()
    for col in annot.columns:
        annot[col] = annot[col].apply(lambda x: f"{x:.0f}%" if pd.notna(x) else "")
    fig = go.Figure(data=go.Heatmap(
        z=cd2.values, x=cd2.columns.tolist(), y=cd2.index.tolist(),
        text=annot.values, texttemplate="%{text}", textfont={"size": 11},
        colorscale="RdYlGn", zmin=0, zmax=100, colorbar=dict(title="Retention %"),
    ))
    fig.update_layout(title="Customer Retention Cohort Matrix (%)", height=500, template="plotly_white",
                      xaxis_title="Months Since First Purchase", yaxis_title="Cohort (First Purchase Month)")
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 5: Geographic Insights
# ============================================================
with tab5:
    st.header("Geographic Sales Insights")
    civ = df.groupby(["city", "state", "region", "lat", "lon"]).agg(
        Revenue=("total_amount", "sum"), Orders=("order_id", "count"), Customers=("customer_id", "nunique"),
    ).reset_index().sort_values("Revenue", ascending=False)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter_geo(civ, lat="lat", lon="lon", size="Revenue", color="region", hover_name="city",
                             hover_data={"Revenue": True, "Orders": True, "Customers": True, "lat": False, "lon": False},
                             title="Revenue Distribution by City", projection="natural earth", size_max=45,
                             color_discrete_sequence=px.colors.qualitative.Set2, scope="asia",
                             center={"lat": 22.5, "lon": 79})
        fig.update_geos(fitbounds="locations")
        fig.update_layout(height=480, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        rr2 = civ.groupby("region").agg(Revenue=("Revenue", "sum"), Orders=("Orders", "sum"),
                                         Cities=("city", "nunique")).reset_index().sort_values("Revenue", ascending=False)
        colors5 = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=rr2["region"], y=rr2["Revenue"] / 1e6,
                             marker_color=colors5[:len(rr2)],
                             text=(rr2["Revenue"] / 1e6).round(1),
                             texttemplate="INR %{text}M", textposition="outside"))
        fig.update_layout(title="Revenue by Region", height=400, template="plotly_white", yaxis_title="Revenue (Millions INR)")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Cities Performance")
    c1, c2 = st.columns(2)
    tc3 = civ.head(10)
    with c1:
        fig = px.bar(tc3, x="Revenue", y="city", orientation="h", color="Revenue", color_continuous_scale="Blues",
                     title="Top 10 Cities by Revenue",
                     text=tc3["Revenue"].apply(lambda x: f"INR {x/1e6:.1f}M"),
                     labels={"Revenue": "Revenue (INR)", "city": ""})
        fig.update_traces(textposition="outside")
        fig.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(tc3, x="Orders", y="city", orientation="h", color="Orders", color_continuous_scale="Greens",
                     title="Top 10 Cities by Orders", text="Orders", labels={"Orders": "Order Count", "city": ""})
        fig.update_traces(textposition="outside")
        fig.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(civ.head(15).style.format({"Revenue": "INR {:,.0f}", "Orders": "{:,}", "Customers": "{:,}"}),
                 use_container_width=True, hide_index=True)

# ============================================================
# TAB 6: RFM Segmentation
# ============================================================
with tab6:
    st.header("RFM Customer Segmentation")
    c1, c2, c3 = st.columns(3)
    with c1:
        sc3 = rfm_seg["Segment"].value_counts()
        fig = px.pie(values=sc3.values, names=sc3.index, title="Customer Segments", hole=0.4,
                     color=sc3.index, color_discrete_map=seg_colors)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(height=420, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        sr2 = rfm_seg.groupby("Segment").agg(Customers=("customer_id", "count"), Total_Revenue=("Monetary", "sum")).reset_index().sort_values("Total_Revenue", ascending=False)
        fig = px.bar(sr2, x="Segment", y="Total_Revenue", title="Revenue by Segment", color="Segment",
                     color_discrete_map=seg_colors, text=sr2["Total_Revenue"].apply(lambda x: f"INR {x/1e6:.1f}M"))
        fig.update_traces(textposition="outside")
        fig.update_layout(height=420, template="plotly_white", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c3:
        sn2 = min(800, len(rfm_seg))
        fig = px.scatter(rfm_seg.sample(sn2), x="Recency", y="Monetary", color="Segment", size="Frequency",
                         title="RFM Customer Map", color_discrete_map=seg_colors,
                         labels={"Recency": "Days Since Last Order", "Monetary": "Total Spend (INR)"},
                         size_max=18, opacity=0.7)
        fig.update_layout(height=420, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Segment Summary Table")
    ss = rfm_seg.groupby("Segment").agg(
        Customers=("customer_id", "count"), Avg_Recency=("Recency", "mean"),
        Avg_Frequency=("Frequency", "mean"), Avg_Monetary=("Monetary", "mean"),
        Total_Revenue=("Monetary", "sum"),
    ).round(0).sort_values("Total_Revenue", ascending=False)
    st.dataframe(ss.style.format({
        "Customers": "{:,}", "Avg_Recency": "{:.0f} days", "Avg_Frequency": "{:.1f}",
        "Avg_Monetary": "INR {:,.0f}", "Total_Revenue": "INR {:,.0f}",
    }), use_container_width=True)

    st.subheader("Segment Recommendations")
    with st.expander("Champions & Loyal Customers"):
        st.markdown("Reward with VIP perks, early access, and referral programs. These drive the most revenue.")
    with st.expander("At Risk & Lost Customers"):
        st.markdown("Launch personalized win-back campaigns with time-limited discounts. Identify and fix churn reasons.")
    with st.expander("New Customers & Need Attention"):
        st.markdown("Strengthen onboarding with welcome offers and product education. Drive toward second purchase.")
