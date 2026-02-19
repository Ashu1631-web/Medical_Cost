import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================
# ✅ STREAMLIT CLOUD SAFE PATH FIX
# =========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(BASE_DIR, "dataset", "medical_insurance.csv")
model_path = os.path.join(BASE_DIR, "model", "best_model.pkl")

# Load Dataset
df = pd.read_csv(csv_path)

# Load Model
model = joblib.load(model_path)

# =========================================
# ✅ Encode Categorical Columns
# =========================================
df["sex"] = df["sex"].map({"male": 0, "female": 1})
df["smoker"] = df["smoker"].map({"no": 0, "yes": 1})
df["region"] = df["region"].map({
    "southwest": 0,
    "southeast": 1,
    "northwest": 2,
    "northeast": 3
})

# =========================================
# ✅ Streamlit Page Setup
# =========================================
st.set_page_config(page_title="Medical Insurance Prediction", layout="wide")

st.title("💰 Medical Insurance Cost Prediction App")
st.write("Enter details to predict medical insurance cost.")

# =========================================
# ✅ Sidebar Inputs
# =========================================
st.sidebar.header("Enter User Information")

age = st.sidebar.slider("Age", 18, 65, 25)
sex = st.sidebar.selectbox("Gender", ["male", "female"])
bmi = st.sidebar.slider("BMI", 15.0, 45.0, 25.0)
children = st.sidebar.selectbox("Children", [0, 1, 2, 3, 4, 5])
smoker = st.sidebar.selectbox("Smoker", ["yes", "no"])
region = st.sidebar.selectbox(
    "Region", ["southwest", "southeast", "northwest", "northeast"]
)

# Encode Input
sex_val = 0 if sex == "male" else 1
smoker_val = 1 if smoker == "yes" else 0
region_val = {"southwest": 0, "southeast": 1, "northwest": 2, "northeast": 3}[region]

# =========================================
# ✅ Prediction Button
# =========================================
if st.sidebar.button("Predict Insurance Cost"):

    input_data = np.array([[age, sex_val, bmi, children, smoker_val, region_val]])
    prediction = model.predict(input_data)[0]

    st.success(f"✅ Estimated Insurance Cost: ₹ {prediction:,.2f}")

# =========================================
# 📊 EDA Dashboard (15 Graphs)
# =========================================
st.header("📊 Exploratory Data Analysis (15 Graphs)")

# Graph 1
st.subheader("1. Charges Distribution")
fig = plt.figure()
sns.histplot(df["charges"], kde=True)
st.pyplot(fig)

# Graph 2
st.subheader("2. Age Distribution")
fig = plt.figure()
sns.histplot(df["age"], kde=True)
st.pyplot(fig)

# Graph 3
st.subheader("3. BMI Distribution")
fig = plt.figure()
sns.histplot(df["bmi"], kde=True)
st.pyplot(fig)

# Graph 4
st.subheader("4. Smoker Count")
fig = plt.figure()
sns.countplot(x=df["smoker"])
st.pyplot(fig)

# Graph 5
st.subheader("5. Gender Count")
fig = plt.figure()
sns.countplot(x=df["sex"])
st.pyplot(fig)

# Graph 6
st.subheader("6. Region Count")
fig = plt.figure()
sns.countplot(x=df["region"])
st.pyplot(fig)

# Graph 7
st.subheader("7. Charges vs Age")
fig = plt.figure()
sns.scatterplot(x="age", y="charges", data=df)
st.pyplot(fig)

# Graph 8
st.subheader("8. Charges vs BMI")
fig = plt.figure()
sns.scatterplot(x="bmi", y="charges", data=df)
st.pyplot(fig)

# Graph 9
st.subheader("9. Charges vs Children")
fig = plt.figure()
sns.boxplot(x="children", y="charges", data=df)
st.pyplot(fig)

# Graph 10
st.subheader("10. Charges by Smoking")
fig = plt.figure()
sns.boxplot(x="smoker", y="charges", data=df)
st.pyplot(fig)

# Graph 11
st.subheader("11. Charges by Gender")
fig = plt.figure()
sns.boxplot(x="sex", y="charges", data=df)
st.pyplot(fig)

# Graph 12
st.subheader("12. Charges by Region")
fig = plt.figure()
sns.boxplot(x="region", y="charges", data=df)
st.pyplot(fig)

# Graph 13
st.subheader("13. Correlation Heatmap")
fig = plt.figure(figsize=(8, 5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
st.pyplot(fig)

# Graph 14
st.subheader("14. Charges Outliers")
fig = plt.figure()
sns.boxplot(y=df["charges"])
st.pyplot(fig)

# Graph 15
st.subheader("15. Age vs Charges (Smoker Effect)")
fig = plt.figure()
sns.scatterplot(x="age", y="charges", hue="smoker", data=df)
st.pyplot(fig)

st.success("✅ All Graphs Loaded Successfully!")
