import pandas as pd
import numpy as np
import pytest

from heartfailure_model.processing.features import (
    Mapper,
    OutlierHandler,
    SexOneHotEncoder,
    ColumnDropper
)


def test_mapper_transformer():
    df = pd.DataFrame({'ST_Slope': ['Flat', 'Up', 'Down', 'Unknown']})
    mapping = {'Up': 0, 'Flat': 1, 'Down': 2}
    transformer = Mapper(variables='ST_Slope', mappings=mapping)
    result = transformer.fit_transform(df)

    # Expected: Unknown → -1
    assert result['ST_Slope'].tolist() == [1, 0, 2, -1]
    assert result['ST_Slope'].dtype == int


def test_outlier_handler():
    df = pd.DataFrame({'Cholesterol': [100, 200, 300, 400, 5000]})
    transformer = OutlierHandler(variables=['Cholesterol'])
    result = transformer.fit_transform(df)

    # 5000 should be capped
    assert result['Cholesterol'].max() < 5000
    assert isinstance(result, pd.DataFrame)


def test_sex_onehot_encoder():
    df = pd.DataFrame({'sex': ['M', 'F', 'M', 'F']})
    transformer = SexOneHotEncoder(variable='sex')
    result = transformer.fit_transform(df)

    assert 'sex_M' in result.columns
    assert 'sex_F' in result.columns
    assert result['sex_M'].sum() == 2
    assert result['sex_F'].sum() == 2
    assert 'sex' not in result.columns


def test_column_dropper():
    df = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
    transformer = ColumnDropper(columns=['B'])
    result = transformer.fit_transform(df)

    assert 'B' not in result.columns
    assert 'A' in result.columns
    assert isinstance(result, pd.DataFrame)