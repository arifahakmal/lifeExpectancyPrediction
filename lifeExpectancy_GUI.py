import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from catboost import CatBoostRegressor
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

# Load data
data_path = "lifeExpectancy_cleaned.csv"  # Update this if the path changes
life_expectancy_data = load_data(data_path)

# Define Features and Target
X = life_expectancy_data.drop(columns=["lifeExpectancy"])
y = life_expectancy_data["lifeExpectancy"]

# Separate Categorical and Numerical Columns
categorical_features = ["Country"]
numerical_features = [
    "Year", "childMortalityRate", "GDP_per_Capita", "totalPopulationPoverty",
    "birthRate", "percentage_of_totalUrbanPopulation",
    "percentageUsage_of_safely_managed_drinking_water", "totalHealthcareExpenditure"
]

# Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# Final Pipeline
model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", CatBoostRegressor(random_state=42, verbose=0))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_pipeline.fit(X_train, y_train)

# Evaluate model
y_pred = model_pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse**0.5
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Streamlit UI
st.title("🌍 Altos Labs Life Expectancy Prediction App")
st.markdown("Predict life expectancy and explore the dataset as well as the model performance.")

# Sidebar for Dataset Exploration
st.sidebar.header("Dataset Exploration")
if st.sidebar.checkbox("Show Dataset"):
    st.write("### Dataset Preview")
    st.dataframe(life_expectancy_data.head())

    st.write("### Dataset Summary")
    st.write(life_expectancy_data.describe())

# Input Fields for Prediction
st.write("## Predict Life Expectancy")

def user_input_features():
    st.write("### Input Features")
    col1, col2, col3 = st.columns(3)

    with col1:
        country = st.selectbox("Country", X["Country"].unique())
        year = st.slider("Year", int(X["Year"].min()), int(X["Year"].max()), int(X["Year"].mean()))
        child_mortality = st.number_input("Child Mortality Rate", value=float(X["childMortalityRate"].mean()), step=0.1)

    with col2:
        gdp = st.number_input("GDP per Capita", value=float(X["GDP_per_Capita"].mean()), step=50.0)
        total_population_poverty = st.number_input(
            "Total Population in Poverty", value=float(X["totalPopulationPoverty"].mean()), step=0.1
        )
        birth_rate = st.number_input("Birth Rate", value=float(X["birthRate"].mean()), step=0.1)

    with col3:
        urban_population = st.number_input(
            "Percentage of Urban Population", value=float(X["percentage_of_totalUrbanPopulation"].mean()), step=0.1
        )
        safe_drinking_water = st.number_input(
            "Percentage Usage of Safe Drinking Water",
            value=float(X["percentageUsage_of_safely_managed_drinking_water"].mean()),
            step=0.1,
        )
        healthcare_exp = st.number_input(
            "Healthcare Expenditure", value=float(X["totalHealthcareExpenditure"].mean()), step=0.1
        )

    inputs = {
        "Country": country,
        "Year": year,
        "childMortalityRate": child_mortality,
        "GDP_per_Capita": gdp,
        "totalPopulationPoverty": total_population_poverty,
        "birthRate": birth_rate,
        "percentage_of_totalUrbanPopulation": urban_population,
        "percentageUsage_of_safely_managed_drinking_water": safe_drinking_water,
        "totalHealthcareExpenditure": healthcare_exp,
    }
    return pd.DataFrame([inputs])

input_df = user_input_features()

# Prediction Button
if st.button("Predict Life Expectancy"):
    with st.spinner("Predicting..."):
        prediction = model_pipeline.predict(input_df)
        st.success(f"Predicted Life Expectancy: {prediction[0]:.2f} years")

    # Visualize Prediction
    st.write("### Predicted vs. Average Life Expectancy")
    avg_life_expectancy = life_expectancy_data["lifeExpectancy"].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(["Predicted"], [prediction[0]], color="blue", label="Predicted")
    ax.bar(["Average"], [avg_life_expectancy], color="green", label="Average")
    ax.set_ylabel("Life Expectancy (Years)")
    ax.legend()
    st.pyplot(fig)

# Model Performance Section
st.write("## Model Performance Metrics")
col1, col2, col3, col4 = st.columns(4)

col1.metric(label="Mean Squared Error", value=f"{mse:.2f}")
col2.metric(label="Root Mean Squared Error", value=f"{rmse:.2f}")
col3.metric(label="Mean Absolute Error", value=f"{mae:.2f}")
col4.metric(label="R² Score", value=f"{r2:.2f}")


