import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from io import BytesIO

st.title("🏡 Airbnb Price Predictor – San Diego")

# ------------------------------
# Load Model from Google Drive
# ------------------------------
@st.cache_resource
def load_model():
    # ⬇️ Replace YOUR_FILE_ID_HERE with the real Google Drive file ID
    url = "https://drive.google.com/uc?export=download&id=1QMvBdSAZzPSLc2PE9SfN4lSOXlyHLy7R"

    response = requests.get(url)
    model_bytes = BytesIO(response.content)
    model = joblib.load(model_bytes)
    return model

model = load_model()


# ------------------------------
# USER INPUTS
# ------------------------------
st.header("Enter Listing Details")

bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, step=1)
bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=10.0, step=0.5)
accommodates = st.number_input("Accommodates", min_value=1, max_value=16, step=1)
minimum_nights = st.number_input("Minimum Nights", min_value=1, max_value=365, step=1)
number_of_reviews = st.number_input("Number of Reviews", min_value=0, max_value=5000, step=1)
review_scores_rating = st.number_input("Review Score Rating (1–100)", min_value=1.0, max_value=100.0, step=1.0)
latitude = st.number_input("Latitude", format="%.6f")
longitude = st.number_input("Longitude", format="%.6f")

room_type = st.selectbox("Room Type", [
    "Entire home/apt",
    "Private room",
    "Shared room",
    "Hotel room"
])

neighbourhood = st.text_input("Neighborhood")  # <-- renamed!


# ------------------------------
# Build Model Input
# ------------------------------
def encode_inputs():
    data = {
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "accommodates": accommodates,
        "minimum_nights": minimum_nights,
        "number_of_reviews": number_of_reviews,
        "review_scores_rating": review_scores_rating,
        "latitude": latitude,
        "longitude": longitude,
        "room_type": room_type,
        "neighbourhood": neighbourhood   # <-- renamed here too!
    }

    df = pd.DataFrame([data])

    # One-hot encode the categorical fields
    df = pd.get_dummies(df, columns=["room_type", "neighbourhood"], drop_first=False)

    # Add missing columns so order matches model.training
    for col in model.feature_names_in_:
        if col not in df.columns:
            df[col] = 0

    df = df[model.feature_names_in_]
    return df


# ------------------------------
# Predict
# ------------------------------
if st.button("Predict Price"):
    X = encode_inputs()
    prediction = model.predict(X)[0]
    st.success(f"Estimated Nightly Price: **${prediction:,.2f}**")
