
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ==========================================================
# 1. PATH CONFIGURATION
# ==========================================================

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data" / "ai4i2020.csv"


# ==========================================================
# 2. FASTAPI APPLICATION
# ==========================================================

app = FastAPI(
    title="Industrial Machine Failure Prediction API",
    description=(
        "Machine failure analysis using Bayes Rule, "
        "Logistic Regression, PCA, Hidden Markov Model "
        "and Random Forest."
    ),
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ==========================================================
# 3. MACHINE SENSOR INPUT
# ==========================================================


from pydantic import BaseModel, Field
from typing import Literal

class MachineInput(BaseModel):
    machine_type: Literal["L", "M", "H"]
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    tool_wear: float



# ==========================================================
# 4. LOAD TRAINED MODELS
# ==========================================================

def load_model(filename):

    model_path = MODEL_DIR / filename

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    return joblib.load(model_path)


try:

    logistic_model = load_model(
        "logistic_regression_pipeline.joblib"
    )

    pca_artifact = load_model(
        "pca_pipeline.joblib"
    )

    rf_artifact = load_model(
        "random_forest_pipeline.joblib"
    )

    hmm_artifact = load_model(
        "hmm_pipeline.joblib"
    )

    models_loaded = True
    model_loading_error = None

except Exception as error:

    models_loaded = False
    model_loading_error = str(error)

    logistic_model = None
    pca_artifact = None
    rf_artifact = None
    hmm_artifact = None


# ==========================================================
# 5. LOAD DATASET FOR BAYES ANALYSIS
# ==========================================================

bayes_statistics = None
bayes_loading_error = None

try:

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    # Use the same stratified training split as the
    # Logistic Regression and Random Forest experiments.

    from sklearn.model_selection import train_test_split

    train_df, _ = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["Machine failure"]
    )

    sensor_mapping = {
        "air_temperature": "Air temperature [K]",
        "process_temperature": "Process temperature [K]",
        "rotational_speed": "Rotational speed [rpm]",
        "torque": "Torque [Nm]",
        "tool_wear": "Tool wear [min]"
    }

    bayes_statistics = {}

    prior_failure = float(
        train_df["Machine failure"].mean()
    )

    for feature, column in sensor_mapping.items():

        threshold = float(
            train_df[column].quantile(0.75)
        )

        high_data = train_df[
            train_df[column] > threshold
        ]

        normal_data = train_df[
            train_df[column] <= threshold
        ]

        bayes_statistics[feature] = {
            "threshold": threshold,

            "high_probability": float(
                high_data["Machine failure"].mean()
            ),

            "normal_probability": float(
                normal_data["Machine failure"].mean()
            ),

            "high_sample_count": len(high_data),

            "normal_sample_count": len(normal_data)
        }

except Exception as error:

    bayes_loading_error = str(error)


# ==========================================================
# 6. FEATURE ENGINEERING
# ==========================================================

def prepare_features(machine):

    mechanical_power = (
        machine.torque
        * machine.rotational_speed
        * (2 * np.pi / 60)
    )

    temperature_difference = (
        machine.process_temperature
        - machine.air_temperature
    )

    wear_torque_interaction = (
        machine.tool_wear * machine.torque
    )

    features = pd.DataFrame([{

        "type": machine.machine_type,

        "air_temperature": machine.air_temperature,

        "process_temperature": machine.process_temperature,

        "rotational_speed": machine.rotational_speed,

        "torque": machine.torque,

        "tool_wear": machine.tool_wear,

        "temperature_difference": temperature_difference,

        "mechanical_power": mechanical_power,

        "wear_torque_interaction": wear_torque_interaction

    }])

    return features


# ==========================================================
# 7. RISK CLASSIFICATION
# ==========================================================

def classify_risk(probability):

    # Initial demonstration thresholds.
    # These must be tuned using validation data.

    if probability < 0.30:
        return "LOW"

    elif probability < 0.70:
        return "MEDIUM"

    else:
        return "HIGH"


# ==========================================================
# 8. UNIT 1 - BAYES RULE
# ==========================================================

def calculate_bayes_analysis(machine):

    if bayes_statistics is None:

        raise HTTPException(
            status_code=503,
            detail=f"Bayes statistics unavailable: {bayes_loading_error}"
        )

    entered_values = {
        "air_temperature": machine.air_temperature,
        "process_temperature": machine.process_temperature,
        "rotational_speed": machine.rotational_speed,
        "torque": machine.torque,
        "tool_wear": machine.tool_wear
    }

    results = {}

    for feature, value in entered_values.items():

        statistics = bayes_statistics[feature]

        threshold = statistics["threshold"]

        is_high = value > threshold

        if is_high:

            probability = statistics["high_probability"]

        else:

            probability = statistics["normal_probability"]

        results[feature] = {

            "entered_value": value,

            "threshold": round(threshold, 2),

            "condition": (
                "HIGH" if is_high else "NORMAL"
            ),

            "failure_probability": round(
                probability, 4
            ),

            "risk_level": classify_risk(probability)

        }

    return {

        "prior_failure_probability": round(
            prior_failure, 4
        ),

        "sensor_condition_analysis": results,

        "method": (
            "Empirical conditional probabilities "
            "estimated from the training dataset."
        )

    }


# ==========================================================
# 9. UNIT 2 - LOGISTIC REGRESSION
# ==========================================================

def calculate_logistic_prediction(X):

    probability = float(
        logistic_model.predict_proba(X)[0, 1]
    )

    prediction = int(
        logistic_model.predict(X)[0]
    )

    return {

        "predicted_failure": prediction,

        "failure_probability": round(
            probability, 4
        ),

        "risk_level": classify_risk(probability)

    }


# ==========================================================
# 10. UNIT 3 - PCA
# ==========================================================

def calculate_pca_prediction(X):

    preprocessor = pca_artifact["preprocessor"]

    pca = pca_artifact["pca"]

    model = pca_artifact["model"]

    processed = preprocessor.transform(X)

    transformed = pca.transform(processed)

    probability = float(
        model.predict_proba(transformed)[0, 1]
    )

    prediction = int(
        model.predict(transformed)[0]
    )

    return {

        "predicted_failure": prediction,

        "failure_probability": round(
            probability, 4
        ),

        "risk_level": classify_risk(probability),

        "principal_components": int(
            pca.n_components_
        ),

        "variance_retained": round(
            float(np.sum(pca.explained_variance_ratio_)),
            4
        )

    }


# ==========================================================
# 11. UNIT 4 - HIDDEN MARKOV MODEL
# ==========================================================

def calculate_hmm_analysis(machine):

    sensor_features = hmm_artifact["sensor_features"]

    scaler = hmm_artifact["scaler"]

    model = hmm_artifact["hmm_model"]

    values = pd.DataFrame([{

        feature: getattr(machine, feature)

        for feature in sensor_features

    }])

    scaled_values = scaler.transform(values)

    # Posterior probabilities of the hidden states
    # for this single sensor observation.

    state_probabilities = model.predict_proba(
        scaled_values
    )[0]

    most_likely_state = int(
        np.argmax(state_probabilities)
    )

    return {

        "most_likely_state": most_likely_state,

        "state_probabilities": {

            f"State {i}": round(
                float(probability), 4
            )

            for i, probability in enumerate(
                state_probabilities
            )

        },

        "interpretation": (
            "Single-observation hidden-state inference. "
            "This does not establish machine degradation "
            "or a temporal transition."
        )

    }


# ==========================================================
# 12. UNIT 5 - RANDOM FOREST
# ==========================================================

def calculate_random_forest_prediction(X):

    preprocessor = rf_artifact["preprocessor"]

    model = rf_artifact["model"]

    processed = preprocessor.transform(X)

    probability = float(
        model.predict_proba(processed)[0, 1]
    )

    prediction = int(
        model.predict(processed)[0]
    )

    return {

        "predicted_failure": prediction,

        "failure_probability": round(
            probability, 4
        ),

        "risk_level": classify_risk(probability)

    }


# ==========================================================
# 13. HOME ENDPOINT
# ==========================================================

@app.get("/")
def home():

    return {

        "application": "Industrial Machine Failure Prediction",

        "status": "running",

        "models_loaded": models_loaded

    }


# ==========================================================
# 14. HEALTH ENDPOINT
# ==========================================================

@app.get("/health")
def health():

    return {

        "status": (
            "healthy" if models_loaded
            and bayes_statistics is not None
            else "degraded"
        ),

        "models_loaded": models_loaded,

        "bayes_statistics_loaded": (
            bayes_statistics is not None
        ),

        "model_loading_error": model_loading_error,

        "bayes_loading_error": bayes_loading_error

    }


# ==========================================================
# 15. MAIN PREDICTION ENDPOINT
# ==========================================================

@app.post("/predict")
def predict_machine_failure(machine: MachineInput):

    if not models_loaded:

        raise HTTPException(
            status_code=503,
            detail=f"Model loading failed: {model_loading_error}"
        )

    if bayes_statistics is None:

        raise HTTPException(
            status_code=503,
            detail=f"Bayes analysis unavailable: {bayes_loading_error}"
        )

    try:

        X = prepare_features(machine)

        # UNIT 1: BAYES RULE

        bayes_results = calculate_bayes_analysis(
            machine
        )

        # UNIT 2: LOGISTIC REGRESSION

        logistic_results = calculate_logistic_prediction(
            X
        )

        # UNIT 3: PCA

        pca_results = calculate_pca_prediction(
            X
        )

        # UNIT 4: HIDDEN MARKOV MODEL

        hmm_results = calculate_hmm_analysis(
            machine
        )

        # UNIT 5: RANDOM FOREST

        rf_results = calculate_random_forest_prediction(
            X
        )

        # COMPLETE RESPONSE

        return {

            "status": "success",

            "machine_input": machine.model_dump(),

            "results": {

                "bayes_rule": bayes_results,

                "logistic_regression": logistic_results,

                "pca": pca_results,

                "hidden_markov_model": hmm_results,

                "random_forest": rf_results

            },

            "disclaimer": (
                "Predictions are based on the synthetic AI4I "
                "dataset. Probability estimates are not "
                "calibrated real-world failure probabilities. "
                "HMM output represents a single-observation "
                "hidden-state inference."
            )

        }

    except HTTPException:

        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )
