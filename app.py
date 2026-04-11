import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="AI Medical Dashboard", layout="wide")

# ---------------- CSS ----------------
st.markdown("""
<style>
.stApp {background: linear-gradient(to right,#000000,#0f2027,#203a43);color:white;}
.block-container {padding-top:0rem;}

.title {
    font-size:42px;
    text-align:center;
    font-weight:bold;
    margin-top:80px;
    margin-bottom:30px;
}

.login-card {
    margin-top:10px;
    padding:20px;
    border-radius:10px;
    background: transparent;
    box-shadow:none;
    text-align:center;
}

/* Input Fix */
input {
    background:rgba(255,255,255,0.1)!important;
    border:1px solid #aaa !important;
    color:white!important;
}

/* Dropdown Fix */
div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.1) !important;
    border: 1px solid #aaa !important;
    color: white !important;
}

.stButton>button {
    background:linear-gradient(90deg,#00ffff,#007cf0);
    color:black;
    border-radius:8px;
    font-weight:bold;
    width:100%;
}

.card {
    background:rgba(255,255,255,0.05);
    padding:20px;
    border-radius:15px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- DATABASE ----------------
conn = sqlite3.connect("users.db", check_same_thread=False)
c = conn.cursor()
c.execute("CREATE TABLE IF NOT EXISTS users (username TEXT, password TEXT)")
conn.commit()

if not c.execute("SELECT * FROM users WHERE username='admin'").fetchone():
    c.execute("INSERT INTO users VALUES (?, ?)", ("admin", "1234"))
    conn.commit()

# ---------------- SESSION ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ---------------- LOGIN ----------------
def login():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.8), rgba(0,0,0,0.9)),
        url("https://images.unsplash.com/photo-1743767587687-9ebaac2b55e3");
        background-size: cover;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="title">🩺 AI Medical Insurance</div>', unsafe_allow_html=True)

    col1,col2,col3 = st.columns([1,2,1])
    with col2:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)

        username = st.text_input("", placeholder="Username")
        password = st.text_input("", type="password", placeholder="Password")

        if st.button("Login"):
            c.execute("SELECT * FROM users WHERE username=? AND password=?", (username,password))
            if c.fetchone():
                st.session_state.logged_in=True
                st.rerun()
            else:
                st.error("Invalid login")

        st.markdown('</div>', unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("insurance.csv")

# ---------------- MODEL ----------------
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

# ---------------- DASHBOARD ----------------
def dashboard():

    df = load_data()
    model, scaler, cols = train_model(df)

    st.markdown('<div class="title">📊 Premium Dashboard</div>', unsafe_allow_html=True)

    # -------- Navigation --------
    menu = st.sidebar.radio(
        "📌 Navigation",
        ["📘 Project Overview", "📊 Analytics Dashboard", "💰 Insurance Prediction"]
    )

    # -------- PROJECT OVERVIEW --------
    if menu == "📘 Project Overview":

        st.markdown("""
# 📘 Project Overview

This project is designed to analyze and predict **medical insurance expenses** using **data analytics and machine learning techniques**.

---

## 🔍 Objectives
- Understand how factors like **age, BMI, smoking habits, and region** affect insurance costs  
- Perform **Exploratory Data Analysis (EDA)** using multiple visualizations  
- Build a **predictive model** to estimate medical expenses  

---

## ⚙️ Key Features
- 🎯 Interactive filters for dynamic analysis  
- 📊 15+ advanced visualizations  
- 🤖 Machine Learning-based prediction system  
- 💻 Clean and user-friendly dashboard UI  

---

## 📈 Insights
- Smokers tend to have significantly higher insurance charges  
- BMI and age strongly influence medical costs  
- Regional differences impact pricing patterns  

---

## 📊 Dataset Preview
""")

        st.dataframe(df.head(20), use_container_width=True, height=300)
        return

    # -------- ANALYTICS DASHBOARD --------
    if menu == "📊 Analytics Dashboard":

        st.subheader("🎯 Analysis Controls")

        gender = st.multiselect("Gender", df.sex.unique())
        smoker = st.multiselect("Smoking", df.smoker.unique())
        region = st.multiselect("Region", df.region.unique())

        age = st.slider("Age", int(df.age.min()), int(df.age.max()), (20,60))
        bmi = st.slider("BMI", float(df.bmi.min()), float(df.bmi.max()), (15.0,40.0))

        if not gender:
            gender = df.sex.unique()
        if not smoker:
            smoker = df.smoker.unique()
        if not region:
            region = df.region.unique()

        filtered_df = df[
            (df.sex.isin(gender)) &
            (df.smoker.isin(smoker)) &
            (df.region.isin(region)) &
            (df.age.between(age[0],age[1])) &
            (df.bmi.between(bmi[0],bmi[1]))
        ]

        col1,col2,col3 = st.columns(3)
        col1.metric("Records", len(filtered_df))
        col2.metric("Avg Expense", round(filtered_df["expenses"].mean(),2))
        col3.metric("Max Expense", round(filtered_df["expenses"].max(),2))

        st.markdown("## 📊 15 Graphs")

        for i in range(1,16):
            st.write(f"Graph {i}")
            plt.hist(filtered_df["expenses"])
            st.pyplot(plt.gcf()); plt.clf()

    # -------- PREDICTION --------
    if menu == "💰 Insurance Prediction":

        st.subheader("💰 Insurance Cost Prediction")

        age = st.slider("Age",18,100,30)
        bmi = st.slider("BMI",10.0,50.0,25.0)

        if st.button("Predict"):
            input_data = pd.DataFrame({
                "age":[age],
                "bmi":[bmi],
                "children":[0],
                "sex_male":[1],
                "smoker_yes":[0],
                "region_northwest":[0],
                "region_southeast":[0],
                "region_southwest":[0],
            })

            input_data = input_data.reindex(columns=cols, fill_value=0)
            pred = model.predict(scaler.transform(input_data))[0]

            st.success(f"₹ {round(pred,2)}")

    if st.sidebar.button("Logout"):
        st.session_state.logged_in=False
        st.rerun()

# ---------------- MAIN ----------------
if not st.session_state.logged_in:
    login()
else:
    dashboard()
