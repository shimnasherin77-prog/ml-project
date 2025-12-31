from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Create FastAPI app
app = FastAPI(
    title="Diamond Price Prediction API",
    description="Predict diamond price based on carat value",
    version="1.0"
)

# Load trained model
model = joblib.load("diamond_price_model.pkl")

# Input data schema
class DiamondInput(BaseModel):
    carat: float

# Root endpoint
@app.get("/")
def home():
    return {"message": "Diamond Price Prediction API is running"}

# Prediction endpoint
@app.post("/predict")
def predict_price(data: DiamondInput):
    input_df = pd.DataFrame([[data.carat]], columns=["carat"])
    prediction = model.predict(input_df)

    return {
        "carat": data.carat,
        "predicted_price": round(float(prediction[0]), 2)
    }
