import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import json
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Body
from fastapi.encoders import jsonable_encoder
from heartfailure_model import __version__ as model_version
from heartfailure_model.predict import make_prediction

from app import __version__, schemas
from app.config import settings

api_router = APIRouter()


@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """
    Root Get
    """
    health = schemas.Health(
        name=settings.PROJECT_NAME, api_version=__version__, model_version=model_version
    )

    return health.dict()



example_input = {
    "inputs": [
        {
            "age": 79,
            "sex": "M",
            "chestpaintype": "ATA",
            "restingbp": 140,
            "cholesterol": 250,
            "fastingbs": 70,
            "restingecg": "Normal",
            "maxhr": 150,
            "exerciseangina": "N",
            "oldpeak": 2.0,
            "st_slope": "Flat",
        }
    ]
}


@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: schemas.MultipleDataInputs = Body(..., example=example_input)) -> Any:
    """
    Heart failure prediction with the heartfailure_model
    """
    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))
    
    results = make_prediction(input_data=input_df.replace({np.nan: None}))

    if results["errors"] is not None:
        raise HTTPException(status_code=400, detail=json.loads(results["errors"]))

    # Convert float predictions to int (binary classification: 0 or 1)
    #results["predictions"] = [int(round(pred)) for pred in results["predictions"]]
    results["predictions"] = int(round(results["predictions"][0]))

    return results
