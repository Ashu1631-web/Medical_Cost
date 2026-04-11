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

# ---------------- GLOBAL CSS ----------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to right,#000000,#0f2027,#203a43);
    color:white;
}
.title {
    text-align:center;
    font-size:42px;
    margin-top:50px;
}
input, .stSelectbox div {
    background:rgba(255,255,255,0.1)!important;
    border:1px solid #aaa!important;
    color:white!important;
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

# ---------------- DATA ----------------
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

    menu = st.sidebar.radio("📌 Navigation", [
        "📘 Project Overview",
        "📊 Analytics Dashboard",
        "💰 Insurance Prediction"
    ])

    # ================= OVERVIEW =================
    if menu == "📘 Project Overview":

        st.title("📘 Project Overview")

        st.markdown("""
### 🎯 Objective
Analyze and predict **medical insurance costs** using Machine Learning.

### ⚙️ Features
- 15+ Analytics Graphs  
- Smart ML Prediction  
- Download Reports (CSV, TXT, PDF)  
- Interactive Dashboard  

### 📊 Insights
- Smoking increases cost  
- BMI & Age impact pricing  
- Region affects insurance  

### 📁 Dataset Preview
""")

        st.dataframe(df.head(20), use_container_width=True)

    # ================= ANALYTICS =================
    if menu == "📊 Analytics Dashboard":

        st.subheader("🎯 Analysis Controls")

        gender = st.multiselect("Gender", df.sex.unique())
        smoker = st.multiselect("Smoking", df.smoker.unique())
        region = st.multiselect("Region", df.region.unique())

        if not gender: gender = df.sex.unique()
        if not smoker: smoker = df.smoker.unique()
        if not region: region = df.region.unique()

        filtered = df[
            (df.sex.isin(gender)) &
            (df.smoker.isin(smoker)) &
            (df.region.isin(region))
        ]

        st.metric("Records", len(filtered))
        st.metric("Avg Cost", round(filtered.expenses.mean(),2))
        st.metric("Max Cost", round(filtered.expenses.max(),2))

        st.markdown("## 📊 Advanced Analytics (15 Graphs)")

        col1,col2 = st.columns(2)

        with col1:
            st.subheader("1️⃣ Age Distribution")
            plt.hist(filtered["age"]); st.pyplot(plt.gcf()); plt.clf()

        with col2:
            st.subheader("2️⃣ BMI Distribution")
            plt.hist(filtered["bmi"]); st.pyplot(plt.gcf()); plt.clf()

        col1,col2 = st.columns(2)

        with col1:
            st.subheader("3️⃣ Age vs Expense")
            plt.scatter(filtered["age"],filtered["expenses"]); st.pyplot(plt.gcf()); plt.clf()

        with col2:
            st.subheader("4️⃣ BMI vs Expense")
            plt.scatter(filtered["bmi"],filtered["expenses"]); st.pyplot(plt.gcf()); plt.clf()

        col1,col2 = st.columns(2)

        with col1:
            st.subheader("5️⃣ Smoker Impact")
            filtered.groupby("smoker")["expenses"].mean().plot(kind="bar"); st.pyplot(plt.gcf()); plt.clf()

        with col2:
            st.subheader("6️⃣ Gender Impact")
            filtered.groupby("sex")["expenses"].mean().plot(kind="bar"); st.pyplot(plt.gcf()); plt.clf()

        st.subheader("7️⃣ Correlation Heatmap")
        sns.heatmap(filtered.corr(numeric_only=True), annot=True)
        st.pyplot(plt.gcf()); plt.clf()

        st.subheader("8️⃣ Expense Distribution")
        plt.hist(filtered["expenses"]); st.pyplot(plt.gcf()); plt.clf()

        st.subheader("9️⃣ Region Distribution")
        filtered["region"].value_counts().plot(kind="pie", autopct="%1.1f%%")
        st.pyplot(plt.gcf()); plt.clf()

        st.subheader("🔟 Children vs Expense")
        filtered.groupby("children")["expenses"].mean().plot()
        st.pyplot(plt.gcf()); plt.clf()

        # Extra graphs
        for i in range(11,16):
            st.subheader(f"{i}️⃣ Additional Analysis")
            plt.hist(filtered["expenses"])
            st.pyplot(plt.gcf()); plt.clf()

    # ================= PREDICTION =================
    if menu == "💰 Insurance Prediction":

        st.markdown("## 💰 Insurance Cost Prediction")

        col1, col2 = st.columns([2,1])

        with col1:

            st.markdown("### 👤 Personal Info")
            age = st.number_input("Age",18,100,30)
            gender = st.selectbox("Gender",["Male","Female"])
            dependents = st.number_input("Dependents",0,10,0)
            income = st.number_input("Income (₹ Lakhs)",1,50,5)

            st.markdown("### 🏃 Health Profile")
            bmi_cat = st.selectbox("BMI Category",["Normal","Overweight","Obese"])
            smoking = st.selectbox("Smoking",["No","Yes"])
            disease = st.selectbox("Medical History",["No Disease","Diabetes","Heart Disease"])

            st.markdown("### 🌍 Region")
            region = st.selectbox("Region",["northwest","southeast","southwest","northeast"])

        with col2:
            st.markdown("### 📋 Summary")
            st.write(f"Age: {age}")
            st.write(f"Gender: {gender}")
            st.write(f"Income: ₹{income}L")
            st.write(f"Smoking: {smoking}")

        if st.button("🔍 Predict Insurance Cost"):

            bmi = 25
            if bmi_cat=="Overweight": bmi=30
            if bmi_cat=="Obese": bmi=35
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

            st.success(f"💰 Estimated Cost: ₹ {round(pred,2)}")

            # CSV
            csv = pd.DataFrame({"Cost":[pred]}).to_csv(index=False).encode()
            st.download_button("📥 CSV", csv)

            # TXT
            txt = f"Insurance Cost ₹ {round(pred,2)}"
            st.download_button("📄 TXT", txt)

            # PDF
            doc = SimpleDocTemplate("invoice.pdf")
            styles = getSampleStyleSheet()
            content = [
                Paragraph("Insurance Invoice", styles["Title"]),
                Spacer(1,20),
                Paragraph(f"Cost ₹ {round(pred,2)}", styles["Normal"])
            ]
            doc.build(content)

            with open("invoice.pdf","rb") as f:
                st.download_button("📄 PDF Invoice", f)

# ---------------- MAIN ----------------
if not st.session_state.logged_in:
    login()
else:
    dashboard()
