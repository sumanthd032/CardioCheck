import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# --- 1. Application Setup ---
# Create a FastAPI app instance
app = FastAPI(title="CardioCheck API")

# Mount the 'frontend' directory to serve static files
# The path '../frontend' is relative to the 'backend' directory where you run uvicorn
app.mount("/static", StaticFiles(directory="../frontend"), name="static")


# --- 2. Load Model and Columns ---
# Load the pre-trained model and the column list on startup.
# This is more efficient as it's done only once.
try:
    # The path is relative to the 'backend' directory
    model = joblib.load('models/logistic_regression_model.joblib')
    model_columns = joblib.load('models/model_columns.joblib')
    print("Model and columns loaded successfully.")
except Exception as e:
    print(f"Error loading model or columns: {e}")
    model = None
    model_columns = None


# --- 3. Define the Input Data Model ---
# Pydantic model to define the structure and data types for incoming request data.
# These fields match the original columns of the dataset before preprocessing.
class PatientData(BaseModel):
    age: int
    sex: str
    cp: str
    trestbps: float
    chol: float
    fbs: str
    restecg: str
    thalch: float
    exang: str
    oldpeak: float
    slope: str
    ca: float
    thal: str


# --- 4. API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """
    Serves the main frontend page.
    """
    # The path is relative to the 'backend' directory
    with open("../frontend/index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.post("/api/predict")
def predict_disease(data: PatientData):
    """
    Prediction endpoint. Takes patient data and returns the heart disease risk.
    """
    if not model or not model_columns:
        return {"error": "Model not loaded. Please check server logs."}

    try:
        # Convert incoming data into a pandas DataFrame
        input_data = pd.DataFrame([data.dict()])

        # One-hot encode the categorical features
        # This must be consistent with the training process
        input_data = pd.get_dummies(input_data)

        # Align the columns of the input data with the model's columns
        # This adds missing columns (with value 0) and removes extra columns
        input_data = input_data.reindex(columns=model_columns, fill_value=0)

        # Make prediction
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)

        # Determine the risk level
        risk = "High Risk" if prediction[0] == 1 else "Low Risk"
        
        # Return the result
        return {
            "prediction": risk,
            "probability": f"{probability[0][1] * 100:.2f}%" # Probability of disease
        }
    except Exception as e:
        return {"error": f"An error occurred during prediction: {str(e)}"}

@app.get("/api/health")
def health_check():
    """
    A simple health check endpoint.
    """
    return {"status": "ok"}
