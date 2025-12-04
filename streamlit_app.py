import streamlit as st
import pandas as pd
import joblib

# Load model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

st.title("🏡 Airbnb Price Predictor – San Diego")
st.write("Enter listing details below to estimate nightly price.")

# Inputs
bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, step=1)
bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=10.0, step=0.5)
accommodates = st.number_input("Accommodates", min_value=1, max_value=16, step=1)
minimum_nights = st.number_input("Minimum Nights", min_value=1, max_value=365, step=1)
number_of_reviews = st.number_input("Number of Reviews", min_value=0, max_value=5000, step=1)
review_scores_rating = st.number_input("Review Score (1–100)", min_value=1.0, max_value=100.0, step=1.0)
latitude = st.number_input("Latitude", format="%.6f")
longitude = st.number_input("Longitude", format="%.6f")

room_type = st.selectbox(
    "Room Type",
    ["Entire home/apt", "Private room", "Shared room", "Hotel room"]
)

neighborhood = st.text_input("Neighborhood (type any value)")

# Build dataframe for model
def make_input_df():
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
        "neighborhood": neighborhood
    }
    return pd.DataFrame([data])

# Predict
if st.button("Predict Nightly Price"):
    X = make_input_df()
    prediction = model.predict(X)[0]
    st.success(f"Estimated Price: **${prediction:,.2f}**")
