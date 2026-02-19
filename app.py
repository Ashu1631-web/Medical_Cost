import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Medical Insurance Cost Prediction", layout="wide")

# ===========================
# SAFE FILE LOADER (NO ERROR)
# ===========================
def find_file(filename):
    for root, dirs, files in os.walk("."):
        if filename in files:
            return os.path.join(root, filename)
    return None

csv_file = find_file("medical_insurance.csv")
model_file = find_file("best_model.pkl")

if csv_file is None:
    st.error("❌ medical_insurance.csv not found. Please upload it inside dataset folder.")
    st.stop()

if model_file is None:
    st.error("❌ best_model.pkl not found. Please upload it inside model folder.")
    st.stop()

df = pd.read_csv(csv_file)
model = joblib.load(model_file)

# Encode categorical columns
df["sex"] = df["sex"].map({"male": 0, "female": 1})
df["smoker"] = df["smoker"].map({"no": 0, "yes": 1})
df["region"] = df["region"].map({
    "southwest": 0,
    "southeast": 1,
    "northwest": 2,
    "northeast": 3
})

st.title("💰 Medical Insurance Cost Prediction App")
st.write("Enter details to predict insurance charges.")

# Sidebar Inputs
st.sidebar.header("User Input Features")

age = st.sidebar.slider("Age", 18, 65, 25)
sex = st.sidebar.selectbox("Gender", ["male", "female"])
bmi = st.sidebar.slider("BMI", 15.0, 45.0, 25.0)
children = st.sidebar.selectbox("Children", [0, 1, 2, 3, 4, 5])
smoker = st.sidebar.selectbox("Smoker", ["yes", "no"])
region = st.sidebar.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# Encode Inputs
sex_val = 0 if sex == "male" else 1
smoker_val = 1 if smoker == "yes" else 0
region_val = {"southwest": 0, "southeast": 1, "northwest": 2, "northeast": 3}[region]

# Prediction
if st.sidebar.button("Predict Cost"):
    input_data = np.array([[age, sex_val, bmi, children, smoker_val, region_val]])
    prediction = model.predict(input_data)[0]
    st.success(f"✅ Estimated Insurance Cost: ₹ {prediction:,.2f}")

# ===========================
# 15 EDA GRAPHS
# ===========================
st.header("📊 Exploratory Data Analysis (15 Graphs)")

graphs = [
    ("Charges Distribution", lambda: sns.histplot(df["charges"], kde=True)),
    ("Age Distribution", lambda: sns.histplot(df["age"], kde=True)),
    ("BMI Distribution", lambda: sns.histplot(df["bmi"], kde=True)),
    ("Smoker Count", lambda: sns.countplot(x=df["smoker"])),
    ("Gender Count", lambda: sns.countplot(x=df["sex"])),
    ("Region Count", lambda: sns.countplot(x=df["region"])),
    ("Charges vs Age", lambda: sns.scatterplot(x="age", y="charges", data=df)),
    ("Charges vs BMI", lambda: sns.scatterplot(x="bmi", y="charges", data=df)),
    ("Charges vs Children", lambda: sns.boxplot(x="children", y="charges", data=df)),
    ("Charges by Smoking", lambda: sns.boxplot(x="smoker", y="charges", data=df)),
    ("Charges by Gender", lambda: sns.boxplot(x="sex", y="charges", data=df)),
    ("Charges by Region", lambda: sns.boxplot(x="region", y="charges", data=df)),
    ("Correlation Heatmap", lambda: sns.heatmap(df.corr(), annot=True)),
    ("Charges Outliers", lambda: sns.boxplot(y=df["charges"])),
    ("BMI Outliers", lambda: sns.boxplot(y=df["bmi"]))
]

for i, (title, plot_func) in enumerate(graphs, 1):
    st.subheader(f"{i}. {title}")
    fig = plt.figure()
    plot_func()
    st.pyplot(fig)

st.success("✅ App Running Successfully with Model + Dataset!")
