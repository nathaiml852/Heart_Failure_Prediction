import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from heartfailure_model.config.core import config
from heartfailure_model.processing.features import Mapper
from heartfailure_model.processing.features import OutlierHandler
from heartfailure_model.processing.features import SexOneHotEncoder
from heartfailure_model.processing.features import ColumnDropper


heartfailure_pipe = Pipeline([
    # **Encoding categorical variables**
    ('sex_encoder', SexOneHotEncoder(variable=config.model_config_.sex_var)),

    # **Mapping categorical variables to numerical values**
    ('map_chestpaintype', Mapper(str(config.model_config_.chestpaintype_var), config.model_config_.chestpaintype_mappings)),

    ('map_restingecg', Mapper(str(config.model_config_.restingecg_var), config.model_config_.restingecg_mappings)),

    ('map_exerciseangina', Mapper(str(config.model_config_.exerciseangina_var), config.model_config_.exerciseangina_mappings)),
    
    ('map_st_slope', Mapper(config.model_config_.st_slope_var, config.model_config_.st_slope_mappings)),
    
    # **Outlier Handling**
    #('outlier_handler', OutlierHandler(variables=['restingbp', 'maxhr', 'oldpeak'])),

    # **Scaling numerical features**
    ('scaler', StandardScaler()),

    # **Model Training**
    ('model_rf', RandomForestClassifier(n_estimators=config.model_config_.n_estimators, 
                                        max_depth=config.model_config_.max_depth,
                                        random_state=config.model_config_.random_state))

])
