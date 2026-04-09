import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="AI Medical Dashboard", layout="wide")

# ----------------------------
# SESSION
# ----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ----------------------------
# LOGIN PAGE (MEDICAL UI)
# ----------------------------
def login():

    st.markdown("""
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1588776814546-ec7e9b4dcf6c");
        background-size: cover;
    }
    .glass {
        background: rgba(0,0,0,0.6);
        padding: 30px;
        border-radius: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align:center;color:white;'>🩺 AI Medical Insurance</h1>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        st.markdown('<div class="glass">', unsafe_allow_html=True)

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if username == "admin" and password == "1234":
                st.success("Welcome Doctor 👨‍⚕️")
                time.sleep(1)
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid Credentials")

        st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# LOAD DATA
# ----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("insurance.csv")

# ----------------------------
# MODEL
# ----------------------------
@st.cache_resource
def train_model(df):
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop("expenses", axis=1)
    y = df["expenses"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    return model, scaler, X.columns

# ----------------------------
# DASHBOARD
# ----------------------------
def dashboard():

    df = load_data()
    model, scaler, cols = train_model(df)

    # ----------------------------
    # SIDEBAR FILTERS (DYNAMIC)
    # ----------------------------
    st.sidebar.title("🔎 Smart Filters")

    gender = st.sidebar.multiselect("Gender", df.sex.unique(), default=df.sex.unique())
    smoker = st.sidebar.multiselect("Smoking Status", df.smoker.unique(), default=df.smoker.unique())
    region = st.sidebar.multiselect("Region", df.region.unique(), default=df.region.unique())

    age = st.sidebar.slider("Age Range", int(df.age.min()), int(df.age.max()), (20, 60))
    bmi = st.sidebar.slider("BMI Range", float(df.bmi.min()), float(df.bmi.max()), (15.0, 40.0))

    # APPLY FILTER
    filtered_df = df[
        (df.sex.isin(gender)) &
        (df.smoker.isin(smoker)) &
        (df.region.isin(region)) &
        (df.age.between(age[0], age[1])) &
        (df.bmi.between(bmi[0], bmi[1]))
    ]

    # ----------------------------
    # HEADER
    # ----------------------------
    st.title("📊 AI Insurance Intelligence Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Patients", len(filtered_df))
    col2.metric("Avg Expense", f"₹ {round(filtered_df['expenses'].mean(),2)}")
    col3.metric("Max Expense", f"₹ {round(filtered_df['expenses'].max(),2)}")

    st.markdown("---")

    # ----------------------------
    # ANIMATION
    # ----------------------------
    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.003)
        progress.progress(i+1)

    st.success("Insights Loaded 🚀")

    # ----------------------------
    # GRAPHS (AUTO UPDATE WITH FILTERS)
    # ----------------------------
    col1, col2 = st.columns(2)

    # 1 Histogram
    with col1:
        st.subheader("Age Distribution")
        fig = plt.figure()
        plt.hist(filtered_df["age"])
        st.pyplot(fig)

    # 2 Histogram BMI
    with col2:
        st.subheader("BMI Distribution")
        fig = plt.figure()
        plt.hist(filtered_df["bmi"])
        st.pyplot(fig)

    # 3 Gender Count
    with col1:
        st.subheader("Gender Distribution")
        fig = plt.figure()
        filtered_df["sex"].value_counts().plot(kind="bar")
        st.pyplot(fig)

    # 4 Smoking
    with col2:
        st.subheader("Smoking Analysis")
        fig = plt.figure()
        filtered_df["smoker"].value_counts().plot(kind="bar")
        st.pyplot(fig)

    # 5 Region
    with col1:
        st.subheader("Region Analysis")
        fig = plt.figure()
        filtered_df["region"].value_counts().plot(kind="bar")
        st.pyplot(fig)

    # 6 Scatter
    with col2:
        st.subheader("BMI vs Expenses")
        fig = plt.figure()
        plt.scatter(filtered_df["bmi"], filtered_df["expenses"])
        st.pyplot(fig)

    # 7 Boxplot
    with col1:
        st.subheader("Expenses by Gender")
        fig = plt.figure()
        filtered_df.boxplot(column="expenses", by="sex")
        st.pyplot(fig)

    # 8 Boxplot
    with col2:
        st.subheader("Expenses by Smoking")
        fig = plt.figure()
        filtered_df.boxplot(column="expenses", by="smoker")
        st.pyplot(fig)

    # 9 Pie
    with col1:
        st.subheader("Region Share")
        fig = plt.figure()
        filtered_df["region"].value_counts().plot(kind="pie", autopct="%1.1f%%")
        st.pyplot(fig)

    # 10 Heatmap
    with col2:
        st.subheader("Correlation Heatmap")
        fig = plt.figure()
        corr = filtered_df.corr(numeric_only=True)
        plt.imshow(corr)
        plt.colorbar()
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.columns)), corr.columns)
        st.pyplot(fig)

    # ----------------------------
    # ML PREDICTION
    # ----------------------------
    st.markdown("---")
    st.subheader("🤖 Predict Expense")

    age_input = st.slider("Age", 18, 100, 30)
    bmi_input = st.slider("BMI", 10.0, 50.0, 25.0)

    if st.button("Predict"):
        input_data = pd.DataFrame({
            "age":[age_input],
            "bmi":[bmi_input],
            "children":[0],
            "sex_male":[1],
            "smoker_yes":[0],
            "region_northwest":[0],
            "region_southeast":[0],
            "region_southwest":[0],
        })

        input_data = input_data.reindex(columns=cols, fill_value=0)
        input_scaled = scaler.transform(input_data)

        pred = model.predict(input_scaled)[0]
        st.success(f"💰 Estimated Expense: ₹ {round(pred,2)}")

    # ----------------------------
    # DOWNLOAD
    # ----------------------------
    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Filtered Data", csv, "data.csv")

    # LOGOUT
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

# ----------------------------
# MAIN
# ----------------------------
if not st.session_state.logged_in:
    login()
else:
    dashboard()
