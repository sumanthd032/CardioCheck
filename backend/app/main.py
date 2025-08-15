import joblib
import pandas as pd
import traceback
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# --- 1. Application Setup ---
app = FastAPI(title="CardioCheck API")

# --- 2. CORS Middleware ---
# Define the list of allowed origins. This is the key fix.
# Browsers treat http://localhost and http://127.0.0.1 as different origins.
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="../frontend"), name="static")


# --- 3. Load Model and Columns ---
try:
    model = joblib.load('models/logistic_regression_model.joblib')
    model_columns = joblib.load('models/model_columns.joblib')
    print("Model and columns loaded successfully.")
except Exception as e:
    print(f"Error loading model or columns: {e}")
    model = None
    model_columns = None


# --- 4. Define the Input Data Model ---
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


# --- 5. API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("../frontend/index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.post("/api/predict")
def predict_disease(data: PatientData):
    """
    Prediction endpoint with corrected data preprocessing logic.
    """
    # **FIXED**: Check for None explicitly instead of truthiness
    if model is None or model_columns is None:
        return JSONResponse(status_code=503, content={"error": "Model not loaded. Please check server logs."})

    try:
        print("\n--- PREDICTION REQUEST RECEIVED ---")
        print(f"Incoming data: {data.dict()}")

        # 1. Convert incoming data into a pandas DataFrame
        input_df = pd.DataFrame([data.dict()])
        print("\nStep 1: DataFrame created from input.")

        # 2. One-hot encode the categorical features. This is the standard approach.
        input_df = pd.get_dummies(input_df)
        print("\nStep 2: DataFrame after one-hot encoding.")

        # 3. Align the columns of the input data with the model's columns.
        # This is the most critical step. It ensures the DataFrame sent to the model
        # has the exact same structure as the one used for training.
        aligned_df = input_df.reindex(columns=model_columns, fill_value=0)
        print("\nStep 3: DataFrame aligned with model columns.")
        
        # 4. Make prediction
        print("\nStep 4: Making prediction...")
        prediction = model.predict(aligned_df)
        probability = model.predict_proba(aligned_df)
        print(f"Prediction: {prediction[0]}, Probability: {probability[0]}")

        risk = "High Risk" if prediction[0] == 1 else "Low Risk"
        
        print("--- PREDICTION SUCCESSFUL ---\n")
        return {
            "prediction": risk,
            "probability": f"{probability[0][1] * 100:.2f}%"
        }
    except Exception as e:
        print("\n---!!! AN ERROR OCCURRED DURING PREDICTION !!!---")
        traceback.print_exc()
        print("--------------------------------------------------\n")
        return JSONResponse(status_code=500, content={"error": f"An error occurred during prediction logic: {str(e)}"})

@app.get("/api/health")
def health_check():
    return {"status": "ok"}
