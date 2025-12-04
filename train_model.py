import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import joblib

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("listings.csv")

# Clean price column
df["price"] = (
    df["price"]
    .astype(str)
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
    .astype(float)
)

# -----------------------------
# Select features (Option A)
# -----------------------------
features = [
    "bedrooms",
    "bathrooms",
    "accommodates",
    "minimum_nights",
    "number_of_reviews",
    "review_scores_rating",
    "latitude",
    "longitude",
    "room_type",
    "neighborhood"
]

df = df[features + ["price"]].dropna()

X = df[features]
y = df["price"]

# Column types
numeric_features = [
    "bedrooms",
    "bathrooms",
    "accommodates",
    "minimum_nights",
    "number_of_reviews",
    "review_scores_rating",
    "latitude",
    "longitude"
]

categorical_features = [
    "room_type",
    "neighborhood"
]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# Model
model = Pipeline(steps=[
    ("prep", preprocessor),
    ("reg", MLPRegressor(hidden_layer_sizes=(64,), max_iter=500, random_state=42))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE:", rmse)

# Save final model
joblib.dump(model, "model.pkl")
print("Saved model as model.pkl")
