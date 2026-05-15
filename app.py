# =========================================================
# Rossmann Store Sales Prediction Dashboard
# Streamlit Web Application
# =========================================================

# =========================
# Import Libraries
# =========================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# =========================
# Page Configuration
# =========================

st.set_page_config(
    page_title="Rossmann Sales Dashboard",
    page_icon="📈",
    layout="wide"
)

# =========================
# Custom CSS Styling
# =========================

st.markdown(
    """
    <style>

    .main {
        background-color: #F5F7FA;
    }

    .stButton>button {
        background-color: #1F77B4;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 18px;
    }

    .metric-box {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# Dashboard Title
# =========================

st.title("📈 Rossmann Store Sales Prediction Dashboard")

st.write(
    """
    Upload store data CSV file and generate future sales predictions.
    """
)

# =========================
# Load Trained Model
# =========================

model = joblib.load(
    "08-05-2026-13-50-12.pkl"
)

# =========================
# Sidebar Inputs
# =========================

st.sidebar.header("Store Parameters")

store_id = st.sidebar.number_input(
    "Store ID",
    min_value=1,
    value=1
)

promo = st.sidebar.selectbox(
    "Promo",
    [0, 1]
)

holiday = st.sidebar.selectbox(
    "Holiday",
    [0, 1]
)

weekend = st.sidebar.selectbox(
    "Weekend",
    [0, 1]
)

competition_distance = st.sidebar.number_input(
    "Competition Distance",
    min_value=0.0,
    value=500.0
)

customers = st.sidebar.number_input(
    "Customers",
    min_value=0,
    value=500
)

# =========================
# CSV Upload Section
# =========================

uploaded_file = st.file_uploader(
    "📂 Upload CSV File",
    type=["csv"]
)

# =========================
# If CSV Uploaded
# =========================

if uploaded_file is not None:

    # Read CSV File

    data = pd.read_csv(uploaded_file)

    # =========================
    # Display Uploaded Data
    # =========================

    st.subheader("📄 Uploaded Dataset")

    st.dataframe(
        data.head()
    )

    # =========================
    # Date Processing
    # =========================

    data['Date'] = pd.to_datetime(
        data['Date']
    )

    # Create Date Features

    data['Year'] = data['Date'].dt.year

    data['Month'] = data['Date'].dt.month

    data['Day'] = data['Date'].dt.day

    data['WeekOfYear'] = (
        data['Date']
        .dt
        .isocalendar()
        .week
        .astype(int)
    )

    data['Weekend'] = (
        data['Date']
        .dt
        .dayofweek
        .isin([5, 6])
        .astype(int)
    )

    # =========================
    # Fill Sidebar Inputs
    # =========================

    data['Store'] = store_id

    data['Promo'] = promo

    data['CompetitionDistance'] = competition_distance

    data['Customers'] = customers

    data['SchoolHoliday'] = holiday

    # =========================
    # Handle Missing Columns
    # =========================

    if 'StoreType' not in data.columns:
        data['StoreType'] = 'a'

    if 'Assortment' not in data.columns:
        data['Assortment'] = 'a'

    if 'StateHoliday' not in data.columns:
        data['StateHoliday'] = '0'

    if 'DayOfWeek' not in data.columns:
        data['DayOfWeek'] = (
            data['Date']
            .dt
            .dayofweek + 1
        )

    # =========================
    # Encode Categorical Columns
    # =========================

    categorical_cols = [
        'StoreType',
        'Assortment',
        'StateHoliday'
    ]

    for col in categorical_cols:

        data[col] = (
            data[col]
            .astype('category')
            .cat.codes
        )

    # =========================
    # Feature Selection
    # =========================

    features = [

        'Store',
        'DayOfWeek',
        'Promo',
        'SchoolHoliday',
        'CompetitionDistance',
        'StoreType',
        'Assortment',
        'Year',
        'Month',
        'Day',
        'WeekOfYear',
        'Weekend'

    ]

    X = data[features]

    # =========================
    # Make Predictions
    # =========================

    predictions = model.predict(X)

    data['PredictedSales'] = predictions

    # =========================
    # Generate Customer Prediction
    # =========================

    data['PredictedCustomers'] = (
        data['PredictedSales'] / 10
    ).astype(int)

    # =========================
    # KPI Metrics
    # =========================

    st.subheader("📊 Prediction Summary")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Average Sales",
            round(
                data['PredictedSales'].mean(),
                2
            )
        )

    with col2:

        st.metric(
            "Maximum Sales",
            round(
                data['PredictedSales'].max(),
                2
            )
        )

    with col3:

        st.metric(
            "Predicted Customers",
            int(
                data['PredictedCustomers'].mean()
            )
        )

    # =========================
    # Prediction Table
    # =========================

    st.subheader("📋 Prediction Results")

    st.dataframe(

        data[
            [
                'Store',
                'Date',
                'PredictedSales',
                'PredictedCustomers'
            ]
        ]

    )

    # =========================
    # Sales Plot
    # =========================

    st.subheader("📈 Predicted Sales Plot")

    fig, ax = plt.subplots(
        figsize=(14, 5)
    )

    ax.plot(

        data['Date'],
        data['PredictedSales'],

        color='blue',
        marker='o',
        linewidth=2

    )

    ax.set_title(
        "Predicted Sales Over Time"
    )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        "Sales"
    )

    plt.xticks(rotation=45)

    st.pyplot(fig)

    # =========================
    # Customer Plot
    # =========================

    st.subheader("👥 Predicted Customers Plot")

    fig2, ax2 = plt.subplots(
        figsize=(14, 5)
    )

    ax2.plot(

        data['Date'],
        data['PredictedCustomers'],

        color='green',
        marker='o',
        linewidth=2

    )

    ax2.set_title(
        "Predicted Customers Over Time"
    )

    ax2.set_xlabel(
        "Date"
    )

    ax2.set_ylabel(
        "Customers"
    )

    plt.xticks(rotation=45)

    st.pyplot(fig2)

    # =========================
    # Download Prediction CSV
    # =========================

    csv = data.to_csv(
        index=False
    )

    st.download_button(

        label="⬇ Download Prediction CSV",

        data=csv,

        file_name="sales_predictions.csv",

        mime="text/csv"

    )

else:

    st.info(
        "Please upload a CSV file to generate predictions."
    )