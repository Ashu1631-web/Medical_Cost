import streamlit as st
import pandas as pd
import time
import sqlite3
import requests
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="AI Medical Dashboard", layout="wide")

# ----------------------------
# DATABASE SETUP
# ----------------------------
conn = sqlite3.connect("users.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS users (
    username TEXT,
    password TEXT
)
""")
conn.commit()

# default user
c.execute("SELECT * FROM users WHERE username='admin'")
if not c.fetchone():
    c.execute("INSERT INTO users VALUES (?, ?)", ("admin", "1234"))
    conn.commit()

# ----------------------------
# SESSION
# ----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ----------------------------
# LOAD DATA
# ----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("insurance.csv")

# ----------------------------
# TRAIN MODEL
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
# LOGIN PAGE (FINAL FIXED)
# ----------------------------
def login():

    st.markdown("""
    <style>
    .stApp {
        background-image: linear-gradient(rgba(0,0,0,0.85), rgba(0,0,0,0.95)),
        url("https://images.unsplash.com/photo-1588776814546-ec7e9b4dcf6c");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    .title {
        font-size: 48px;
        text-align:center;
        color:white;
        font-weight:bold;
        margin-top:20px;
    }

    .login-card {
        background: rgba(0,0,0,0.6);
        padding: 35px;
        border-radius: 15px;
        backdrop-filter: blur(15px);
        box-shadow: 0 0 20px rgba(0,255,255,0.3);
    }

    input {
        background: rgba(255,255,255,0.1) !important;
        border: 2px solid #00ffff !important;
        border-radius: 10px !important;
        color: white !important;
        box-shadow: 0 0 8px #00ffff;
    }

    label {display:none;}

    .stButton>button {
        background: linear-gradient(90deg, #00ffff, #007cf0);
        color: black;
        border-radius: 10px;
        font-weight: bold;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="title">🩺 AI MEDICAL INSURANCE</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)

        username = st.text_input("", placeholder="👤 Username")
        password = st.text_input("", type="password", placeholder="🔒 Password")

        if st.button("Login"):
            c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
            if c.fetchone():
                st.success("Welcome 🚀")
                time.sleep(1)
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid Credentials")

        st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# DASHBOARD
# ----------------------------
def dashboard():

    df = load_data()
    model, scaler, cols = train_model(df)

    # SIDEBAR FILTERS
    st.sidebar.title("🔎 Filters")

    gender = st.sidebar.multiselect("Gender", df.sex.unique(), default=df.sex.unique())
    smoker = st.sidebar.multiselect("Smoking", df.smoker.unique(), default=df.smoker.unique())
    region = st.sidebar.multiselect("Region", df.region.unique(), default=df.region.unique())

    age = st.sidebar.slider("Age", int(df.age.min()), int(df.age.max()), (20, 60))
    bmi = st.sidebar.slider("BMI", float(df.bmi.min()), float(df.bmi.max()), (15.0, 40.0))

    filtered_df = df[
        (df.sex.isin(gender)) &
        (df.smoker.isin(smoker)) &
        (df.region.isin(region)) &
        (df.age.between(age[0], age[1])) &
        (df.bmi.between(bmi[0], bmi[1]))
    ]

    # HEADER
    st.title("📊 AI Medical Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Records", len(filtered_df))
    col2.metric("Avg Expense", f"₹ {round(filtered_df['expenses'].mean(),2)}")
    col3.metric("Max Expense", f"₹ {round(filtered_df['expenses'].max(),2)}")

    st.markdown("---")

    # GRAPHS
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Age Distribution")
        st.bar_chart(filtered_df["age"].value_counts())

    with col2:
        st.subheader("BMI vs Expense")
        st.scatter_chart(filtered_df[["bmi", "expenses"]])

    # ML PREDICTION
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
        st.success(f"💰 ₹ {round(pred,2)}")

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
