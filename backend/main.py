
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# --------------------------------------------------
# 1. APPLICATION CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"

app = FastAPI(
    title="Industrial Machine Failure Risk API",
    description=(
        "Machine failure prediction using "
        "Logistic Regression, PCA and Random Forest."
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


# --------------------------------------------------
# 2. SENSOR INPUT SCHEMA
# --------------------------------------------------

class MachineInput(BaseModel):

    machine_type: str = Field(
        ...,
        pattern="^[LMH]$",
        description="Machine type: L, M or H"
    )

    air_temperature: float = Field(
        ..., ge=250, le=350
    )

    process_temperature: float = Field(
        ..., ge=250, le=350
    )

    rotational_speed: float = Field(
        ..., gt=0, le=10000
    )

    torque: float = Field(
        ..., ge=0, le=1000
    )

    tool_wear: float = Field(
        ..., ge=0, le=1000
    )


# --------------------------------------------------
# 3. LOAD TRAINED MODELS
# --------------------------------------------------

def load_model(filename):

    path = MODEL_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {path}"
        )

    return joblib.load(path)


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

    models_loaded = True

except Exception as error:

    models_loaded = False
    model_loading_error = str(error)

    logistic_model = None
    pca_artifact = None
    rf_artifact = None


# --------------------------------------------------
# 4. FEATURE ENGINEERING
# --------------------------------------------------

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

    return pd.DataFrame([{
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


# --------------------------------------------------
# 5. RISK CLASSIFICATION
# --------------------------------------------------

def classify_risk(probability):

    # Prototype thresholds only.
    # These must be calibrated using validation data.

    if probability < 0.30:
        return "LOW"

    elif probability < 0.70:
        return "MEDIUM"

    return "HIGH"


# --------------------------------------------------
# 6. HEALTH CHECK
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "application": "Industrial Machine Failure Risk API",
        "status": "running",
        "models_loaded": models_loaded
    }


@app.get("/health")
def health():

    return {
        "status": "healthy" if models_loaded else "degraded",
        "models_loaded": models_loaded,
        "error": None if models_loaded else model_loading_error
    }


# --------------------------------------------------
# 7. PREDICTION ENDPOINT
# --------------------------------------------------

@app.post("/predict")
def predict_machine_failure(machine: MachineInput):

    if not models_loaded:
        raise HTTPException(
            status_code=503,
            detail=f"Model loading failed: {model_loading_error}"
        )

    try:

        X = prepare_features(machine)

        # Logistic Regression
        logistic_probability = float(
            logistic_model.predict_proba(X)[0, 1]
        )

        # PCA + Logistic Regression
        pca_processed = (
            pca_artifact["preprocessor"].transform(X)
        )

        pca_transformed = (
            pca_artifact["pca"].transform(pca_processed)
        )

        pca_probability = float(
            pca_artifact["model"].predict_proba(
                pca_transformed
            )[0, 1]
        )

        # Random Forest
        rf_processed = (
            rf_artifact["preprocessor"].transform(X)
        )

        rf_probability = float(
            rf_artifact["model"].predict_proba(
                rf_processed
            )[0, 1]
        )

        results = {
            "logistic_regression": {
                "failure_probability": round(
                    logistic_probability, 4
                ),
                "risk_level": classify_risk(
                    logistic_probability
                )
            },

            "pca_logistic_regression": {
                "failure_probability": round(
                    pca_probability, 4
                ),
                "risk_level": classify_risk(
                    pca_probability
                )
            },

            "random_forest": {
                "failure_probability": round(
                    rf_probability, 4
                ),
                "risk_level": classify_risk(
                    rf_probability
                )
            }
        }

        return {
            "status": "success",
            "machine_input": machine.model_dump(),
            "predictions": results,
            "note": (
                "Probabilities are model estimates, not "
                "calibrated real-world failure probabilities. "
                "Risk thresholds are provisional."
            )
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )
