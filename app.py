import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(page_title="AI Medical Dashboard", layout="wide")

# ---------------- CSS ----------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to right,#000000,#0f2027,#203a43);
    color:white;
}
.title {text-align:center;font-size:42px;margin-top:80px;}
input, .stSelectbox div {
    background:rgba(255,255,255,0.1)!important;
    border:1px solid #aaa!important;
    color:white!important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- DB ----------------
conn = sqlite3.connect("users.db", check_same_thread=False)
c = conn.cursor()
c.execute("CREATE TABLE IF NOT EXISTS users (username TEXT, password TEXT)")
conn.commit()

if not c.execute("SELECT * FROM users WHERE username='admin'").fetchone():
    c.execute("INSERT INTO users VALUES (?, ?)", ("admin", "1234"))
    conn.commit()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ---------------- LOGIN ----------------
def login():
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(rgba(0,0,0,0.8), rgba(0,0,0,0.9)),
        url("https://images.unsplash.com/photo-1743767587687-9ebaac2b55e3");
        background-size: cover;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="title">🩺 AI Medical Insurance</div>', unsafe_allow_html=True)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (username,password))
        if c.fetchone():
            st.session_state.logged_in=True
            st.rerun()
        else:
            st.error("Invalid login")

# ---------------- LOAD ----------------
@st.cache_data
def load_data():
    return pd.read_csv("insurance.csv")

@st.cache_resource
def train_model(df):
    df = pd.get_dummies(df, drop_first=True)
    X = df.drop("expenses", axis=1)
    y = df["expenses"]

    scaler = StandardScaler()
    model = LinearRegression()
    model.fit(scaler.fit_transform(X), y)

    return model, scaler, X.columns

# ---------------- DASHBOARD ----------------
def dashboard():

    df = load_data()
    model, scaler, cols = train_model(df)

    menu = st.sidebar.radio("📌 Navigation", ["📘 Overview","📊 Analytics","💰 Prediction"])

    # -------- OVERVIEW --------
    if menu == "📘 Overview":
        st.title("📘 Project Overview")
        st.write("""
        This project predicts medical insurance cost using machine learning.
        Includes analytics dashboard + prediction + downloads.
        """)
        st.dataframe(df.head(20))

    # -------- ANALYTICS --------
    if menu == "📊 Analytics":

        gender = st.multiselect("Gender", df.sex.unique())
        smoker = st.multiselect("Smoking", df.smoker.unique())
        region = st.multiselect("Region", df.region.unique())

        if not gender: gender=df.sex.unique()
        if not smoker: smoker=df.smoker.unique()
        if not region: region=df.region.unique()

        filtered = df[(df.sex.isin(gender)) & (df.smoker.isin(smoker)) & (df.region.isin(region))]

        st.metric("Records", len(filtered))
        st.metric("Avg Cost", round(filtered.expenses.mean(),2))

        st.markdown("## 📊 15 Graphs")

        col1,col2 = st.columns(2)
        with col1:
            plt.hist(filtered["age"]); st.pyplot(plt.gcf()); plt.clf()
        with col2:
            plt.hist(filtered["bmi"]); st.pyplot(plt.gcf()); plt.clf()

        col1,col2 = st.columns(2)
        with col1:
            plt.scatter(filtered["age"],filtered["expenses"]); st.pyplot(plt.gcf()); plt.clf()
        with col2:
            plt.scatter(filtered["bmi"],filtered["expenses"]); st.pyplot(plt.gcf()); plt.clf()

        col1,col2 = st.columns(2)
        with col1:
            filtered.groupby("smoker")["expenses"].mean().plot(kind="bar"); st.pyplot(plt.gcf()); plt.clf()
        with col2:
            filtered.groupby("sex")["expenses"].mean().plot(kind="bar"); st.pyplot(plt.gcf()); plt.clf()

        sns.heatmap(filtered.corr(numeric_only=True),annot=True)
        st.pyplot(plt.gcf()); plt.clf()

    # -------- PREDICTION --------
    if menu == "💰 Prediction":

        st.subheader("💰 Insurance Cost Prediction")

        age = st.number_input("Age",18,100,30)
        gender = st.selectbox("Gender",["Male","Female"])
        dependents = st.number_input("Dependents",0,10,0)
        income = st.number_input("Income (Lakhs)",1,50,5)

        bmi_category = st.selectbox("BMI",["Normal","Overweight","Obese"])
        smoking = st.selectbox("Smoking",["No","Yes"])
        disease = st.selectbox("Medical History",["No Disease","Diabetes","Heart Disease"])
        region = st.selectbox("Region",["northwest","southeast","southwest","northeast"])

        if st.button("Predict"):

            bmi = 25
            if bmi_category=="Overweight": bmi=30
            if bmi_category=="Obese": bmi=35
            if smoking=="Yes": bmi+=3
            if disease!="No Disease": bmi+=2

            input_data = pd.DataFrame({
                "age":[age],
                "bmi":[bmi],
                "children":[dependents],
                "sex_male":[1 if gender=="Male" else 0],
                "smoker_yes":[1 if smoking=="Yes" else 0],
                "region_northwest":[1 if region=="northwest" else 0],
                "region_southeast":[1 if region=="southeast" else 0],
                "region_southwest":[1 if region=="southwest" else 0],
            })

            input_data = input_data.reindex(columns=cols, fill_value=0)
            pred = model.predict(scaler.transform(input_data))[0]

            st.success(f"₹ {round(pred,2)}")

            # CSV
            st.download_button("CSV", pd.DataFrame({"Cost":[pred]}).to_csv().encode())

            # TXT
            st.download_button("TXT", f"Cost ₹ {pred}")

            # PDF
            doc = SimpleDocTemplate("invoice.pdf")
            styles = getSampleStyleSheet()
            content=[Paragraph("Invoice",styles["Title"]),
                     Paragraph(f"Cost ₹ {round(pred,2)}",styles["Normal"])]
            doc.build(content)

            with open("invoice.pdf","rb") as f:
                st.download_button("PDF Invoice",f)

# ---------------- MAIN ----------------
if not st.session_state.logged_in:
    login()
else:
    dashboard()
