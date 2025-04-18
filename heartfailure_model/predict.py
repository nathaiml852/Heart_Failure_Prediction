import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from heartfailure_model import __version__ as _version
from heartfailure_model.config.core import config
from heartfailure_model.pipeline import heartfailure_pipe
from heartfailure_model.processing.data_manager import load_pipeline
from heartfailure_model.processing.validation import validate_inputs

# Load the saved pipeline
pipeline_file_name = f"{config.app_config_.pipeline_save_file}{_version}.pkl"
print(f"🔍 Loading pipeline from: {pipeline_file_name}")  # ✅ Debug pipeline path
heartfailure_pipe = load_pipeline(file_name=pipeline_file_name)

if heartfailure_pipe is None:
    raise RuntimeError("🚨 Error: Model pipeline failed to load!")


def make_prediction(*,input_data:Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """

    validated_data, errors = validate_inputs(input_df=pd.DataFrame(input_data))
    
    #validated_data=validated_data.reindex(columns=['Pclass','Sex','Age','Fare', 'Embarked','FamilySize','Has_cabin','Title'])
    validated_data=validated_data.reindex(columns=config.model_config_.features)
    #print(validated_data)
    results = {"predictions": None, "version": _version, "errors": errors}
    
    predictions = heartfailure_pipe.predict(validated_data)

    results = {"predictions": predictions,"version": _version, "errors": errors}
    #print(results)
    import numpy as np
    final_prediction = int(np.round(results["predictions"][0]))
    print(f"💓 Heart Disease Prediction (0 = No, 1 = Yes): {final_prediction}")
    
    if not errors:

        predictions = heartfailure_pipe.predict(validated_data)
        results = {"predictions": predictions,"version": _version, "errors": errors}
        #print(results)

    return results

if __name__ == "__main__":
    data_in = {
    'age': [53],
    'sex': ["M"],
    'chestpaintype': ["ASY"],
    'restingbp': [120],
    'cholesterol': [246],
    'fastingbs': [0],
    'restingecg': ["Normal"],
    'maxhr': [116],
    'exerciseangina': ["Y"],
    'oldpeak': [0],
    'st_slope': ["Flat"]
}
    input_data = pd.DataFrame(data_in)
    print(f"Raw Input Data:\n{pd.DataFrame(input_data)}\n")  # 🟢 Debugging line
    result = make_prediction(input_data=data_in)
    print(result)