import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pytest
import pandas as pd
import numpy as np
from heartfailure_model.config.core import model_config_

@pytest.fixture
def sample_input_data():
    return pd.DataFrame([
        {
            model_config_.age_var: np.nan,
            model_config_.sex_var: "M",
            model_config_.chestpaintype_var: "ATA",
            model_config_.restingbp_var: 130,
            model_config_.cholesterol_var: 250,
            model_config_.fastingbs_var: 1,
            model_config_.restingecg_var: "Normal",
            model_config_.maxhr_var: 150,
            model_config_.exerciseangina_var: "N",
            model_config_.oldpeak_var: 1.0,
            model_config_.st_slope_var: "Up"
        },
        {
            model_config_.age_var: 85,
            model_config_.sex_var: "F",
            model_config_.chestpaintype_var: "NAP",
            model_config_.restingbp_var: 300,
            model_config_.cholesterol_var: np.nan,
            model_config_.fastingbs_var: 0,
            model_config_.restingecg_var: "LVH",
            model_config_.maxhr_var: 60,
            model_config_.exerciseangina_var: "Y",
            model_config_.oldpeak_var: 2.5,
            model_config_.st_slope_var: "Flat"
        }
    ])