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
.title {font-size:42px;text-align:center;font-weight:bold;margin-top:20px;}
.login-card {
    margin-top:40px;
    padding:45px;
    border-radius:20px;
    background:rgba(255,255,255,0.08);
    backdrop-filter: blur(15px);
    box-shadow:0 0 30px rgba(0,255,255,0.2);
    text-align:center;
    transition:0.3s;
}
.login-card:hover {
    transform: scale(1.02);
}
input {
    background:rgba(255,255,255,0.1)!important;
    border:2px solid #00ffff!important;
    color:white!important;
}
label{display:none;}
.stButton>button {
    background:linear-gradient(90deg,#00ffff,#007cf0);
    color:black;
    border-radius:8px;
    font-weight:bold;
    text-align:center;
    font-size:16px;
}
.card {background:rgba(255,255,255,0.05);padding:20px;border-radius:15px;transition:0.3s;}
.card:hover {transform:translateY(-5px);box-shadow:0 0 15px #00ffff;}
div[data-baseweb="select"] > div {border:none!important;box-shadow:none!important;}
div[data-baseweb="tag"] {background:rgba(0,255,255,0.2)!important;border:1px solid #00ffff!important;color:white!important;}
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

    # Background only for login
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.75), rgba(0,0,0,0.9)),
        url("https://images.unsplash.com/photo-1588776814546-ec7e4b3c8c8c");
        background-size: cover;
        background-position: center;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="title">🩺 AI Medical Insurance</div>', unsafe_allow_html=True)

    col1,col2,col3 = st.columns([1,2,1])
    with col2:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)

        # ✅ Remember Me Autofill
        default_user = st.session_state.get("remember_user", "")
        username = st.text_input("", value=default_user, placeholder="👤 Username")

        # ✅ Show/Hide Password
        show_pass = st.checkbox("Show Password")
        password = st.text_input(
            "",
            type="default" if show_pass else "password",
            placeholder="🔒 Password"
        )

        remember = st.checkbox("Remember Me")

        if st.button("🚀 Login"):
            c.execute("SELECT * FROM users WHERE username=? AND password=?", (username,password))
            if c.fetchone():
                st.session_state.logged_in=True

                if remember:
                    st.session_state["remember_user"] = username

                st.success("Login Successful ✅")
                st.rerun()
            else:
                st.error("Invalid login ❌")

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

    # ✅ Reset background after login
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right,#000000,#0f2027,#203a43);
    }
    </style>
    """, unsafe_allow_html=True)

    df = load_data()
    model, scaler, cols = train_model(df)

    st.markdown('<div class="title">📊 Premium Dashboard</div>', unsafe_allow_html=True)

    st.sidebar.title("🎯 Filters")

    gender = st.sidebar.multiselect("Gender", df.sex.unique())
    smoker = st.sidebar.multiselect("Smoking", df.smoker.unique())
    region = st.sidebar.multiselect("Region", df.region.unique())

    age = st.sidebar.slider("Age", int(df.age.min()), int(df.age.max()), (20,60))
    bmi = st.sidebar.slider("BMI", float(df.bmi.min()), float(df.bmi.max()), (15.0,40.0))

    # Overview
    if not gender or not smoker or not region:
        st.warning("Select filters to view analysis")
        return

    filtered_df = df[
        (df.sex.isin(gender)) &
        (df.smoker.isin(smoker)) &
        (df.region.isin(region)) &
        (df.age.between(age[0],age[1])) &
        (df.bmi.between(bmi[0],bmi[1]))
    ]

    # KPI
    col1,col2,col3 = st.columns(3)
    col1.markdown(f'<div class="card">Records<br><h2>{len(filtered_df)}</h2></div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="card">Avg Expense<br><h2>{round(filtered_df["expenses"].mean(),2)}</h2></div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="card">Max Expense<br><h2>{round(filtered_df["expenses"].max(),2)}</h2></div>', unsafe_allow_html=True)

    st.markdown("## 📊 Advanced Analytics")

    col1,col2 = st.columns(2)

    with col1:
        plt.hist(filtered_df["age"])
        st.pyplot(plt.gcf()); plt.clf()

    with col2:
        plt.scatter(filtered_df["bmi"], filtered_df["expenses"])
        st.pyplot(plt.gcf()); plt.clf()

    st.subheader("🤖 Prediction")

    age_input = st.slider("Age",18,100,30)
    bmi_input = st.slider("BMI",10.0,50.0,25.0)

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

        st.success(f"Estimated Expense: {round(pred,2)}")

    if st.sidebar.button("Logout"):
        st.session_state.logged_in=False
        st.rerun()

# ---------------- MAIN ----------------
if not st.session_state.logged_in:
    login()
else:
    dashboard()
