import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import hashlib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ----------------- PAGE CONFIGURATION -----------------
st.set_page_config(page_title="Predictive Healthcare Analytics", layout="wide")

# ----------------- DATABASE & FILE PATH SETUP -----------------
DB_FILE = "vaccination_data.db"
USER_DB = "users.db"
DATASET_PATH = "/Users/pavansappidi/Desktop/TARSS/Tars1/finald.xlsx"

# Function to create database connection
def create_connection(db_path):
    return sqlite3.connect(db_path)

# ----------------- FETCH DATA FROM DATABASE -----------------
def load_data():
    conn = create_connection(DB_FILE)
    df = pd.read_sql("SELECT * FROM vaccination_data", conn)
    conn.close()
    return df

df = load_data()  # Load the data into the dataframe

# ----------------- USER AUTHENTICATION SYSTEM -----------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Function to check if a user exists in the database
def user_exists(username):
    conn = create_connection(USER_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()
    return user

# Function to verify login credentials
def authenticate_user(username, password):
    conn = create_connection(USER_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
    stored_password = cursor.fetchone()
    conn.close()
    return stored_password and stored_password[0] == hash_password(password)

# ----------------- LOGIN SYSTEM -----------------
def login_page():
    st.title("🔑 Secure Login")
    username = st.text_input("👤 Username")
    password = st.text_input("🔑 Password", type="password")
    
    if st.button("Login"):
        if authenticate_user(username, password):
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.rerun()
        else:
            st.error("❌ Invalid credentials. Please try again.")

# ----------------- MAIN DASHBOARD -----------------
st.title("📊 Predictive Healthcare Analytics for Vaccine Administration")

if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
    login_page()
    st.stop()

st.sidebar.header("🔍 Filter Data")
if not df.empty:
    state = st.sidebar.selectbox("📍 Select State", df["STATE"].dropna().unique())
    city = st.sidebar.selectbox("🏙 Select City", df[df["STATE"] == state]["CITY"].dropna().unique())
    selected_vaccine_types = st.sidebar.multiselect("💉 Select Vaccine Type(s)", df["DESCRIPTION"].dropna().unique())

    if selected_vaccine_types:
        filtered_df = df[(df["STATE"] == state) & (df["CITY"] == city) & (df["DESCRIPTION"].isin(selected_vaccine_types))]
        st.write(f"## 📊 Data for {city}, {state} ({', '.join(selected_vaccine_types)})")
        st.dataframe(filtered_df)

        # ----------------- TREND ANALYSIS -----------------
        st.write("### 📊 Vaccination Trends")
        st.plotly_chart(px.pie(filtered_df, names="ETHNICITY", title="Ethnicity Distribution"))
        st.plotly_chart(px.pie(filtered_df, names="GENDER", title="Gender Distribution"))
        st.plotly_chart(px.bar(filtered_df, x="AGE_GROUP", color="VACCINATED", title="Vaccination by Age Group"))
        
        # ----------------- PREDICTION USING RANDOM FOREST -----------------
        if 'Year' in filtered_df.columns and not filtered_df.empty:
            forecast_df = filtered_df.groupby("Year")["VACCINATED"].sum().reset_index()
            forecast_df["Year"] = forecast_df["Year"].astype(int)
            
            if len(forecast_df) > 5:
                X = forecast_df[["Year"]]
                y = forecast_df["VACCINATED"]
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                future_years = pd.DataFrame({"Year": list(range(forecast_df["Year"].max() + 1, forecast_df["Year"].max() + 6))})
                future_predictions = model.predict(future_years)
                
                future_forecast = pd.DataFrame({"Year": future_years["Year"], "VACCINATED": future_predictions})
                combined_forecast = pd.concat([forecast_df, future_forecast])
                
                st.plotly_chart(px.line(combined_forecast, x="Year", y="VACCINATED", title="Future Vaccination Demand Prediction (Random Forest)"))
                st.write(f"Mean Absolute Error of Model: {mean_absolute_error(y_test, model.predict(X_test)):.2f}")
            else:
                st.warning("⚠️ Not enough data points for prediction. Please select a broader dataset.")
