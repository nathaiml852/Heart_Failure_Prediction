from typing import Any, List, Optional

from pydantic import BaseModel


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    #predictions: Optional[List[int]]
    predictions: Optional[int]

class DataInputSchema(BaseModel):
    age: Optional[int]
    sex: Optional[str]
    chestpaintype: Optional[str]
    restingbp: Optional[int]
    cholesterol: Optional[int]
    fastingbs: Optional[int]
    restingecg: Optional[str]
    maxhr: Optional[int]
    exerciseangina: Optional[str]
    oldpeak: Optional[float]
    st_slope: Optional[str]

class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]
