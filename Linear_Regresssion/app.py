
# 1. Import Libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#2.page config
st.set_page_config(page_title="Insurance prediction",layout="centered")
st.title("Medical insurance cost prediction")
st.write("Linear regression model")

#3.load data
@st.cache_data
def load_data():
    return pd.read_csv("insurance.csv")
df=load_data()
st.subheader("dataset preview")
st.dataframe(df.head())

#4.data cleaning
df=df.drop_duplicates()

#5. encoding
le=LabelEncoder()
df['sex']=le.fit_transform(df["sex"])
df['smoker']=le.fit_transform(df["smoker"])
df=pd.get_dummies(df,columns=['region'],drop_first=True)

#6.Features and target
x=df.drop("charges",axis=1)
y=df["charges"]

#7.train_test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#8.scaling
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

#9.model training
model=LinearRegression()
model.fit(x_train,y_train)

#10.prediction
y_pred=model.predict(x_test)

# =========================
# 11. Evaluation
# =========================
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

adj_r2 = 1 - (1 - r2) * (len(y_test)-1) / (len(y_test)-x_test.shape[1]-1)

# =========================
# 12. Visualization
# =========================
st.subheader("Actual vs Predicted")

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.set_xlabel("Actual Charges")
ax.set_ylabel("Predicted Charges")
st.pyplot(fig)

# =========================
# 13. Metrics
# =========================
st.subheader("Model Performance")

c1, c2 = st.columns(2)
c1.metric("MAE", f"{mae:.2f}")
c2.metric("RMSE", f"{rmse:.2f}")

c3, c4 = st.columns(2)
c3.metric("R2 Score", f"{r2:.3f}")
c4.metric("Adjusted R2", f"{adj_r2:.3f}")

# =========================
# 14. User Input
# =========================
st.subheader("Predict Insurance Charges")

age = st.slider("Age", 18, 100, 30)
bmi = st.slider("BMI", 10.0, 50.0, 25.0)
children = st.slider("Children", 0, 5, 0)

sex = st.selectbox("Sex", ["female", "male"])
smoker = st.selectbox("Smoker", ["no", "yes"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Encode Inputs
sex_val = 1 if sex == "male" else 0
smoker_val = 1 if smoker == "yes" else 0

region_northwest = 1 if region == "northwest" else 0
region_southeast = 1 if region == "southeast" else 0
region_southwest = 1 if region == "southwest" else 0

# Final Input
input_data = np.array([[age, sex_val, bmi, children, smoker_val,
                        region_northwest, region_southeast, region_southwest]])

input_scaled = scaler.transform(input_data)

prediction = model.predict(input_scaled)[0]

st.success(f"Predicted Charges: ₹ {prediction:,.2f}")