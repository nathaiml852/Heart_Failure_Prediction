import numpy as np
import pandas as pd
import pytest
from heartfailure_model.processing.features import (
    SexOneHotEncoder,
    OutlierHandler,
    Mapper,
    ColumnDropper,
)
from heartfailure_model.config.core import model_config_

# Sample input data for testing
@pytest.fixture
def sample_input_data():
    return pd.DataFrame({
        'sex': ['M', 'F', 'M', 'F'],
        'stslope': ['Up', 'Down', 'Flat', 'Up'],
        'restingbp': [120, 140, 130, 115],
        'cholesterol': [250, 300, 350, 400],
        'maxhr': [150, 160, 170, 180],
        'oldpeak': [1.2, 2.3, 3.4, 0.5],
        'chestpaintype': ['typical', 'nonanginal', 'asymptomatic', 'nonanginal'],
        'restingecg': ['Normal', 'ST-T Wave Abnormality', 'Normal', 'Left Ventricular Hypertrophy'],
        'exerciseangina': ['N', 'Y', 'Y', 'N'],
    })

def test_sex_encoder(sample_input_data):
    transformer = SexOneHotEncoder(variable=model_config_.sex_var)
    transformed = transformer.fit_transform(sample_input_data.copy())

    # Get expected encoded column names
    expected_columns = ['sex_M', 'sex_F']
    
    # Assert both one-hot columns are present
    for col in expected_columns:
        assert col in transformed.columns
    
    # Check that only 0 or 1 values exist in these columns
    for col in expected_columns:
        assert set(transformed[col].unique()).issubset({0, 1})

def test_chestpain_mapper(sample_input_data):
    mapping = {'typical': 0, 'nonanginal': 1, 'asymptomatic': 2}
    transformer = Mapper(variables=model_config_.chestpaintype_var, mappings=mapping)
    transformed = transformer.fit_transform(sample_input_data.copy())
    assert transformed[model_config_.chestpaintype_var].dtype in [np.int64, np.float64]
    assert set(transformed[model_config_.chestpaintype_var].unique()).issubset({0, 1, 2})

def test_restingecg_mapper(sample_input_data):
    mapping = {'Normal': 0, 'ST-T Wave Abnormality': 1, 'Left Ventricular Hypertrophy': 2}
    transformer = Mapper(variables=model_config_.restingecg_var, mappings=mapping)
    transformed = transformer.fit_transform(sample_input_data.copy())
    assert transformed[model_config_.restingecg_var].dtype in [np.int64, np.float64]
    assert set(transformed[model_config_.restingecg_var].unique()).issubset({0, 1, 2})

def test_exerciseangina_mapper(sample_input_data):
    mapping = {'Y': 1, 'N': 0}
    transformer = Mapper(variables=model_config_.exerciseangina_var, mappings=mapping)
    transformed = transformer.fit_transform(sample_input_data.copy())
    assert set(transformed[model_config_.exerciseangina_var].unique()).issubset({0, 1})

def test_stslope_mapper(sample_input_data):
    mapping = {'Up': 0, 'Flat': 1, 'Down': 2}
    transformer = Mapper(variables=model_config_.st_slope_var, mappings=mapping)
    transformed = transformer.fit_transform(sample_input_data.copy())
    assert transformed[model_config_.st_slope_var].dtype in [np.int64, np.float64]
    assert set(transformed[model_config_.st_slope_var].unique()).issubset({0, 1, 2})

def test_outlier_handler(sample_input_data):
    transformer = OutlierHandler(variables=[
        model_config_.restingbp_var,
        model_config_.cholesterol_var,
        model_config_.maxhr_var,
        model_config_.oldpeak_var
    ])
    transformed = transformer.fit_transform(sample_input_data.copy())
    # Ensure no NaN values after transformation
    assert transformed.isna().sum().sum() == 0

def test_column_dropper(sample_input_data):
    transformer = ColumnDropper(columns=['restingbp', 'cholesterol'])
    transformed = transformer.fit_transform(sample_input_data.copy())
    assert 'restingbp' not in transformed.columns
    assert 'cholesterol' not in transformed.columns
    assert 'sex' in transformed.columns  # Ensure other columns remain