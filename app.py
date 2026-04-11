import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(page_title="AI Medical Dashboard", layout="wide")

# ---------------- CSS ----------------
st.markdown("""
<style>
.stApp {background: linear-gradient(to right,#000000,#0f2027,#203a43);color:white;}
.title {text-align:center;font-size:40px;margin-top:80px;}
.login-card {background:none;box-shadow:none;text-align:center;}
input {background:rgba(255,255,255,0.1)!important;border:1px solid #aaa!important;color:white!important;}
div[data-baseweb="select"] > div {background:rgba(255,255,255,0.1)!important;}
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

# ---------------- PDF FUNCTION ----------------
def generate_pdf(age, gender, income, smoking, region, pred):

    file_path = "insurance_report.pdf"
    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()

    content = []
    content.append(Paragraph("AI Medical Insurance Invoice", styles["Title"]))
    content.append(Spacer(1,20))

    content.append(Paragraph(f"Age: {age}", styles["Normal"]))
    content.append(Paragraph(f"Gender: {gender}", styles["Normal"]))
    content.append(Paragraph(f"Income: ₹ {income} Lakhs", styles["Normal"]))
    content.append(Paragraph(f"Smoking: {smoking}", styles["Normal"]))
    content.append(Paragraph(f"Region: {region}", styles["Normal"]))

    content.append(Spacer(1,20))
    content.append(Paragraph(f"Estimated Cost: ₹ {round(pred,2)}", styles["Heading2"]))

    doc.build(content)
    return file_path

# ---------------- LOGIN ----------------
def login():
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

    menu = st.sidebar.radio("Navigation", ["Overview","Analytics","Prediction"])

    # -------- OVERVIEW --------
    if menu == "Overview":
        st.title("📘 Project Overview")
        st.write("Insurance cost prediction using ML")
        st.dataframe(df.head())

    # -------- ANALYTICS --------
    if menu == "Analytics":

        gender = st.multiselect("Gender", df.sex.unique())
        smoker = st.multiselect("Smoking", df.smoker.unique())
        region = st.multiselect("Region", df.region.unique())

        if not gender: gender=df.sex.unique()
        if not smoker: smoker=df.smoker.unique()
        if not region: region=df.region.unique()

        filtered = df[(df.sex.isin(gender)) & (df.smoker.isin(smoker)) & (df.region.isin(region))]

        st.metric("Records", len(filtered))
        st.metric("Avg Cost", round(filtered.expenses.mean(),2))

        for i in range(15):
            plt.hist(filtered["expenses"])
            st.pyplot(plt.gcf()); plt.clf()

    # -------- PREDICTION --------
    if menu == "Prediction":

        age = st.slider("Age",18,100,30)
        income = st.slider("Income",1,50,5)
        gender = st.selectbox("Gender",["Male","Female"])
        smoking = st.selectbox("Smoking",["Yes","No"])
        region = st.selectbox("Region",["northwest","southeast","southwest","northeast"])

        if st.button("Predict"):

            input_data = pd.DataFrame({
                "age":[age],
                "bmi":[25],
                "children":[0],
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
            df_out = pd.DataFrame({"Cost":[pred]})
            st.download_button("Download CSV", df_out.to_csv().encode(), "result.csv")

            # PDF
            pdf = generate_pdf(age, gender, income, smoking, region, pred)
            with open(pdf,"rb") as f:
                st.download_button("Download PDF Invoice", f, "invoice.pdf")

# ---------------- MAIN ----------------
if not st.session_state.logged_in:
    login()
else:
    dashboard()
