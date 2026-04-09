import streamlit as st
import pandas as pd
import time

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="Pro Dashboard", layout="wide")

# ----------------------------
# SESSION STATE (LOGIN)
# ----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ----------------------------
# LOGIN PAGE
# ----------------------------
def login_page():

    st.markdown("""
        <style>
        .big-title {
            font-size:40px;
            text-align:center;
            font-weight:bold;
            color:#4CAF50;
        }
        .login-box {
            background-color:#111;
            padding:30px;
            border-radius:15px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="big-title">🔐 AI Insurance Dashboard</p>', unsafe_allow_html=True)

    st.write("")

    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        st.markdown('<div class="login-box">', unsafe_allow_html=True)

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if username == "admin" and password == "1234":
                st.success("Login Successful 🚀")
                time.sleep(1)
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid Credentials ❌")

        st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# LOADING ANIMATION
# ----------------------------
def loading_animation():
    with st.spinner("Loading Dashboard..."):
        time.sleep(2)

# ----------------------------
# DASHBOARD
# ----------------------------
def dashboard():

    # Load data
    df = pd.read_csv("insurance.csv")

    # Sidebar
    st.sidebar.title("Filters")

    age = st.sidebar.slider("Age", int(df.age.min()), int(df.age.max()), (20, 60))
    bmi = st.sidebar.slider("BMI", float(df.bmi.min()), float(df.bmi.max()), (15.0, 40.0))
    smoker = st.sidebar.multiselect("Smoker", df.smoker.unique(), default=df.smoker.unique())

    filtered_df = df[
        (df.age.between(age[0], age[1])) &
        (df.bmi.between(bmi[0], bmi[1])) &
        (df.smoker.isin(smoker))
    ]

    # HEADER
    st.title("📊 Insurance Pro Dashboard")

    # KPI
    col1, col2, col3 = st.columns(3)

    col1.metric("Records", len(filtered_df))
    col2.metric("Avg Expense", f"₹ {round(filtered_df['expenses'].mean(),2)}")
    col3.metric("Max Expense", f"₹ {round(filtered_df['expenses'].max(),2)}")

    st.markdown("---")

    # SIMPLE ANIMATION EFFECT
    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.005)
        progress.progress(i + 1)

    st.success("Dashboard Loaded 🎉")

    # CHARTS
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Age Distribution")
        st.bar_chart(filtered_df["age"].value_counts())

    with col2:
        st.subheader("BMI vs Expenses")
        st.scatter_chart(filtered_df[["bmi", "expenses"]])

    # DATA
    st.markdown("### Data Preview")
    st.dataframe(filtered_df.head(20))

    # LOGOUT
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

# ----------------------------
# MAIN FLOW
# ----------------------------
if not st.session_state.logged_in:
    login_page()
else:
    loading_animation()
    dashboard()
