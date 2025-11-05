import sys
import os
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import FunctionTransformer, TargetEncoder, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
sys.path.append(os.path.abspath("../src")) 
from src.features.preprocess import deterministic_fe
from src.features.transformers import PersonalTitleTransformer

def preprocessing_pipeline():
    categorical_transformer= ColumnTransformer([
        ('ohe', OneHotEncoder(handle_unknown='ignore', drop='first',sparse_output=False).set_output(transform = 'pandas'), ['Embarked', 'Personal_title']),
        ('target_enc', TargetEncoder(target_type='binary', random_state=42).set_output(transform = 'pandas'), ['Sex']),
        ('iterative_imputer', IterativeImputer(imputation_order='ascending',
                                            random_state = 42,
                                            min_value=1,
                                            max_value=80,
                                            tol = 0.00001),["Ticket_repeated",
                                                            "Family_size" ,
                                                            "Fare",
                                                            "Parch",
                                                            "SibSp",
                                                            "Pclass","Age"])
        
    ], remainder='passthrough', verbose_feature_names_out= False).set_output(transform="pandas")

    scaling = ColumnTransformer([('normalization', PowerTransformer('yeo-johnson'), ['Fare', 'Pclass', 'SibSp', 'Parch', 'Age','Sex', 'Family_size'])
                        ], remainder='passthrough', verbose_feature_names_out=False).set_output(transform="pandas")

    categorical_preprocessor = Pipeline([
        ('Deterministic_fe', FunctionTransformer(deterministic_fe)), #drop ticket, idPassenger
        ('personal_title', PersonalTitleTransformer(treshold=40)),   #drop Name
        ('encode', categorical_transformer),
        ('scaling', scaling)
        ])
    
    return categorical_preprocessor