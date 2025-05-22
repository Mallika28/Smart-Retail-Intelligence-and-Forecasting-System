import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from utils import load_and_preprocess
from utils_report import create_pdf_report, get_download_link, export_to_excel, get_excel_download_link
from visualizations import create_mrp_plot, create_outlet_size_plot, create_outlet_type_plot, create_correlation_heatmap
from additional_visualizations import create_feature_importance_plot, create_product_category_plot, create_outlet_comparison_plot, create_mrp_sales_scatter
from time_series import generate_time_series_data, create_time_series_plot, forecast_sales
from models.advanced_model import model

# Page Setup
st.set_page_config(page_title="Retail Forecasting Dashboard", layout="wide")

# ============ MODERN UI STYLING ============
st.markdown("""
<style>
body {
    font-family: 'Segoe UI', sans-serif;
    background-color: #a6bfd9;
    color: #152230;
}
h1, h2, h3 {
    color: #1f2937;
}
[data-testid="stMetric"] {
    background-color: #a6bfd9;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    margin: 0.5rem 0;
    text-align: center;
}
.stButton > button {
    background-color: #a6bfd9;
    color: #ffffff;
    font-weight: 600;
    padding: 10px 20px;
    border-radius: 6px;
    border: none;
}
.stButton > button:hover {
    background-color: #1e40af;
}
[data-testid="stTabs"] button {
    color: #1f2937;
    font-weight: 500;
    border-bottom: 2px solid transparent;
}
[data-testid="stTabs"] [aria-selected="true"] {
    border-bottom: 3px solid #2563eb;
    background-color: #e0e7ff;
}
.stDataFrame {
    background-color: white;
    border-radius: 8px;
    font-size: 0.9rem;
}
.stRadio > div {
    flex-direction: row;
}
</style>
""", unsafe_allow_html=True)

# ============ Load Data ============
df = load_and_preprocess("datasets/train.csv")

st.title("Retail Trends & Performance Monitor")

# ============ Tabs ============
tabs = st.tabs([
    "Overview",
    "Advanced Insights",
    "Sales Prediction Tool",
    "Export Report"
])

# ============ Overview ============
with tabs[0]:
    st.title("Overview")

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Sales", f"${df['Item_Outlet_Sales'].mean():,.2f}")
    col2.metric("Products", df['Item_Identifier'].nunique())
    col3.metric("Outlets", df['Outlet_Identifier'].nunique())
    col4.metric("Total Records", len(df))

    # Data Preview
    if st.checkbox("Show Dataset Preview"):
        rows = st.slider("Rows to preview", 5, 50, 10)
        st.dataframe(df.head(rows), use_container_width=True)

    # Basic Sales Insights
    st.subheader("Basic Sales Insights")
    col1, col2 = st.columns(2)
    col1.plotly_chart(create_mrp_plot(df), use_container_width=True)
    col2.plotly_chart(create_outlet_size_plot(df), use_container_width=True)
    col1.plotly_chart(create_outlet_type_plot(df), use_container_width=True)
    col2.plotly_chart(create_correlation_heatmap(df), use_container_width=True)

    # Time Series
    st.subheader("Sales Trend Over Time")
    ts_df = generate_time_series_data(df, start_date=(datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d"), days=365)
    st.plotly_chart(create_time_series_plot(ts_df), use_container_width=True)

# ============ Advanced Insights ============
with tabs[1]:
    st.title("Advanced Insights")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Feature Importance")
        try:
            imp = model.get_feature_importance()
        except:
            imp = {
                "Item_MRP": 0.35,
                "Outlet_Type": 0.25,
                "Outlet_Size": 0.20,
                "Outlet_Age": 0.15,
                "Item_Visibility": 0.05
            }
        st.plotly_chart(create_feature_importance_plot(imp), use_container_width=True)

    with col2:
        st.subheader("Product Categories")
        st.plotly_chart(create_product_category_plot(df), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Price vs Sales")
        st.plotly_chart(create_mrp_sales_scatter(df), use_container_width=True)
    with col2:
        st.subheader("Outlet Performance")
        st.plotly_chart(create_outlet_comparison_plot(df), use_container_width=True)

# ============ Sales Prediction Tool ============
with tabs[2]:
    st.title("Sales Prediction Tool")

    mode = st.radio("Choose Mode", ["Basic", "Advanced"], horizontal=True)

    if mode == "Basic":
        st.markdown("#### Enter Product & Outlet Details")

        item_mrp = st.slider("Item MRP", 50.0, 300.0, 140.0)
        outlet_type = st.selectbox("Outlet Type", ["Grocery Store", "Supermarket Type1", "Supermarket Type2", "Supermarket Type3"])
        outlet_size = st.selectbox("Outlet Size", ["Small", "Medium", "High"])
        est_year = st.slider("Establishment Year", 1985, 2024, 2000)

        outlet_map = {"Grocery Store": 0, "Supermarket Type1": 1, "Supermarket Type2": 2, "Supermarket Type3": 3}
        size_map = {"High": 0, "Medium": 1, "Small": 2}
        outlet_age = 2025 - est_year

        features = np.array([[item_mrp, 1, size_map[outlet_size], outlet_map[outlet_type], outlet_age]])

        if st.button("Predict Sales"):
            prediction = model.predict(features)[0]
            st.success(f"Predicted Sales: ${prediction:,.2f}")

    else:
        st.markdown("#### Upload CSV for Batch Prediction")
        uploaded = st.file_uploader("Upload CSV", type="csv")
        if uploaded:
            try:
                batch = pd.read_csv(uploaded)
                st.dataframe(batch.head(), use_container_width=True)
                st.info("Batch prediction logic can be added here.")
            except Exception as e:
                st.error(f"Error reading file: {e}")

# ============ Export Report ============
with tabs[3]:
    st.title("Export Report")

    format = st.radio("Select Format", ["PDF", "Excel"], horizontal=True)
    with st.expander("Customize Report"):
        pred = st.checkbox("Include Sample Prediction", True)
        feat = st.checkbox("Include Feature Importance", True)

    if st.button("Generate Report"):
        sample_pred = {
            'inputs': {'Item MRP': '$150.00'},
            'prediction': 1500.0,
            'lower_bound': 1200.0,
            'upper_bound': 1800.0
        } if pred else None

        try:
            feat_imp = model.get_feature_importance() if feat else None
        except:
            feat_imp = None

        if format == "PDF":
            pdf = create_pdf_report(df, sample_pred, feat_imp)
            st.markdown(get_download_link(pdf, "Retail_Report.pdf"), unsafe_allow_html=True)
        else:
            excel = export_to_excel(df, sample_pred)
            st.markdown(get_excel_download_link(excel, "Retail_Report.xlsx"), unsafe_allow_html=True)
