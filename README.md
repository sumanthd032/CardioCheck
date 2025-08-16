# CardioCheck : Heart Disease Risk Predictor

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/TailwindCSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)

CardioCheck is a simple, end-to-end web application that predicts a
user's risk of heart disease based on their health metrics. It features
a clean frontend built with HTML and Tailwind CSS, and a powerful
backend powered by FastAPI and a Scikit-learn logistic regression model.

## Features
- **Interactive UI:** A user-friendly web form for
inputting health data. 
- **ML-Powered Predictions:** Utilizes a trained
logistic regression model to predict risk
- **Real-time Results:**
Instantly displays the prediction ("Low Risk" or "High Risk") and the
calculated probability. 
- **RESTful API:** A well-defined API built with
FastAPI to handle prediction requests.

------------------------------------------------------------------------

## Tech Stack
- **Frontend:** HTML, Tailwind CSS, JavaScript
- **Backend:** Python, FastAPI
- **Machine Learning:** Scikit-learn, Pandas, NumPy, Joblib
- **Server:** Uvicorn

------------------------------------------------------------------------

## Getting Started
Follow these instructions to get a copy of the project up and running on
your local machine for development and testing purposes.

### Prerequisites

-   Python 3.10 or higher
-   pip and venv for package management

### Installation

1.  Clone the repository:

    ``` bash
    git clone https://github.com/sumanthd032/CardioCheck
    cd cardiocheck
    ```

2.  Set up the backend virtual environment:

    ``` bash
    cd backend
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  Install the required Python packages:

    ``` bash
    pip install -r requirements.txt
    ```

4.  Download the dataset:

    -   Download the "Heart Disease UCI" dataset from Kaggle.
    -   Unzip the file and find `heart_disease_uci.csv`.
    -   Rename it to `heart.csv`.
    -   Create a `data` folder in the project's root directory and place
        `heart.csv` inside it.

------------------------------------------------------------------------

## Usage

1.  **Train the Model:**

    -   Navigate to the `backend/notebooks` directory.\
    -   Run the `Model_Training.ipynb` Jupyter Notebook. This will
        process the data and save the trained model
        (`logistic_regression_model.joblib`) and column data
        (`model_columns.joblib`) to the `backend/models` directory.

2.  **Start the Backend Server:**

    ``` bash
    cd backend
    uvicorn app.main:app --reload
    ```

    -   The server will be running at <http://127.0.0.1:8000>.

3.  **Launch the Frontend:**

    -   Open your web browser and navigate to <http://127.0.0.1:8000>.
    -   Fill in the form with patient data and click **"Calculate
        Risk"** to see the prediction.

------------------------------------------------------------------------


## Disclaimer
This tool is for educational and informational purposes only and does
not constitute medical advice. The predictions are based on a dataset
and a statistical model and should not be used for self-diagnosis.
Always consult a qualified healthcare professional for any health
concerns.