from typing import List, Optional, Tuple, Union
from datetime import datetime
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from heartfailure_model.config.core import config
#from heartfailure_model.processing.data_manager import pre_pipeline_preparation
import pandas as pd
from typing import Tuple, Any

def validate_inputs(input_df: pd.DataFrame) -> Tuple[pd.DataFrame, Any]:
    """Check model inputs for unprocessable values."""
    
    # Example: You might add checks for nulls or valid ranges
    errors = {}

    if input_df.isnull().any().any():
        errors["missing_values"] = "Input contains missing values."

    # You could return the cleaned data directly or keep original
    return input_df, errors if errors else None


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