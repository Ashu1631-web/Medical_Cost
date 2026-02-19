import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("dataset/medical_insurance.csv")

# Load Model
model = joblib.load("model/best_model.pkl")

# Encode categorical
df["sex"] = df["sex"].map({"male": 0, "female": 1})
df["smoker"] = df["smoker"].map({"no": 0, "yes": 1})
df["region"] = df["region"].map({
    "southwest": 0,
    "southeast": 1,
    "northwest": 2,
    "northeast": 3
})

# Streamlit Config
st.set_page_config(page_title="Medical Insurance Cost Prediction", layout="wide")

st.title("💰 Medical Insurance Cost Prediction App")
st.markdown("Predict insurance charges using ML Regression Models.")

# Sidebar Inputs
st.sidebar.header("Enter User Details")

age = st.sidebar.slider("Age", 18, 65, 25)
sex = st.sidebar.selectbox("Gender", ["male", "female"])
bmi = st.sidebar.slider("BMI", 15.0, 45.0, 25.0)
children = st.sidebar.selectbox("Children", [0, 1, 2, 3, 4, 5])
smoker = st.sidebar.selectbox("Smoker", ["yes", "no"])
region = st.sidebar.selectbox(
    "Region", ["southwest", "southeast", "northwest", "northeast"]
)

# Encode Inputs
sex_val = 0 if sex == "male" else 1
smoker_val = 1 if smoker == "yes" else 0
region_val = {"southwest": 0, "southeast": 1, "northwest": 2, "northeast": 3}[region]

# Prediction Button
if st.sidebar.button("Predict Cost"):

    input_data = np.array([[age, sex_val, bmi, children, smoker_val, region_val]])
    prediction = model.predict(input_data)[0]

    st.success(f"✅ Estimated Insurance Cost: ₹ {prediction:,.2f}")


# ==============================
# 📊 EDA SECTION WITH 15 GRAPHS
# ==============================

st.header("📊 Exploratory Data Analysis Dashboard")

col1, col2 = st.columns(2)

# Graph 1 - Charges Distribution
with col1:
    st.subheader("1. Charges Distribution")
    fig = plt.figure()
    sns.histplot(df["charges"], kde=True)
    st.pyplot(fig)

# Graph 2 - Age Distribution
with col2:
    st.subheader("2. Age Distribution")
    fig = plt.figure()
    sns.histplot(df["age"], kde=True)
    st.pyplot(fig)

# Graph 3 - Smokers Count
st.subheader("3. Smoker vs Non-Smoker Count")
fig = plt.figure()
sns.countplot(x=df["smoker"])
st.pyplot(fig)

# Graph 4 - Gender Count
st.subheader("4. Gender Distribution")
fig = plt.figure()
sns.countplot(x=df["sex"])
st.pyplot(fig)

# Graph 5 - Region Count
st.subheader("5. Region Distribution")
fig = plt.figure()
sns.countplot(x=df["region"])
st.pyplot(fig)

# Graph 6 - Charges vs Age
st.subheader("6. Charges vs Age")
fig = plt.figure()
sns.scatterplot(x="age", y="charges", data=df)
st.pyplot(fig)

# Graph 7 - Charges vs BMI
st.subheader("7. Charges vs BMI")
fig = plt.figure()
sns.scatterplot(x="bmi", y="charges", data=df)
st.pyplot(fig)

# Graph 8 - Charges vs Children
st.subheader("8. Charges vs Children")
fig = plt.figure()
sns.boxplot(x="children", y="charges", data=df)
st.pyplot(fig)

# Graph 9 - Smoker Impact on Charges
st.subheader("9. Charges by Smoking Status")
fig = plt.figure()
sns.boxplot(x="smoker", y="charges", data=df)
st.pyplot(fig)

# Graph 10 - Gender Impact
st.subheader("10. Charges by Gender")
fig = plt.figure()
sns.boxplot(x="sex", y="charges", data=df)
st.pyplot(fig)

# Graph 11 - Region Impact
st.subheader("11. Charges by Region")
fig = plt.figure()
sns.boxplot(x="region", y="charges", data=df)
st.pyplot(fig)

# Graph 12 - Correlation Heatmap
st.subheader("12. Correlation Heatmap")
fig = plt.figure(figsize=(8,5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
st.pyplot(fig)

# Graph 13 - Pairplot
st.subheader("13. Pairplot (Numeric Features)")
st.write("Pairplot not shown fully in Streamlit due to size limitations.")

# Graph 14 - Charges Outliers
st.subheader("14. Outlier Detection (Charges)")
fig = plt.figure()
sns.boxplot(y=df["charges"])
st.pyplot(fig)

# Graph 15 - BMI Outliers
st.subheader("15. Outlier Detection (BMI)")
fig = plt.figure()
sns.boxplot(y=df["bmi"])
st.pyplot(fig)

st.info("✅ Total 15 Graphs Displayed Successfully!")
