import pandas as pd 
import numpy as np
import sys
import os
from sklearn.model_selection import  GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from joblib import dump
sys.path.append(os.path.abspath("../"))
from src.data.load_data import load_dataset
sys.path.append(os.path.abspath("../src")) 
from src.features.pipe_config import preprocessing_pipeline
from src.utils.helpers import grid_search_res
import time



def train_model(X_train, y_train, model):
    preprocess = preprocessing_pipeline()
    if isinstance(model, SVC):
        ex = Pipeline([
        ('preprocessing', preprocess),
        ('standardize', StandardScaler()),
        ('model', model)
        ])
    else:  
        ex = Pipeline([
            ('preprocessing', preprocess),
            ('model', model)
        ])
    
    ex.fit(X_train, y_train)
    return ex


def train_all_models(X_train, y_train, seed) -> dict:
    start_full = time.perf_counter()
    print('Training Logistic model...')
    log_reg_clf = LogisticRegression(C=0.9,
                                    penalty='l1',
                                    solver='liblinear',
                                    random_state=seed)
    log_reg_clf = train_model(X_train, y_train, log_reg_clf)
    end_log = time.perf_counter()

    print(f'Training logistic model took {end_log-start_full} s. \n\n Training Decision tree model...')
    dt_clf = DecisionTreeClassifier(ccp_alpha=0.01,
                                    random_state=seed)
    dt_clf = train_model(X_train, y_train, dt_clf)
    end_df = time.perf_counter()
    print(f'Training took {end_df-end_log:.2f} s. \n\n Training Random forest model...')

    rf_clf = RandomForestClassifier(ccp_alpha=0.003,
                                    max_depth=15,
                                    max_features=0.9, 
                                    min_samples_leaf=2, 
                                    min_samples_split=5,
                                    random_state=seed)
    rf_clf = train_model(X_train, y_train, rf_clf)
    end_rf = time.perf_counter()
    print(f'Training took {end_rf-end_df:.2f} s. \n\n Training Ada boost model...')

    aboost_clf = AdaBoostClassifier(learning_rate=0.12,
                                    n_estimators=600,
                                    random_state=seed)
    aboost_clf = train_model(X_train, y_train, aboost_clf)
    end_ab= time.perf_counter()
    print(f'Training took {end_ab-end_rf:.2f} s. \n\n Training support vector machines model...')
    svc = SVC(kernel='rbf', 
                    C=0.5174678103906237,
                    gamma=0.07096086061283641,
                    random_state=seed)
    svc = train_model(X_train, y_train, svc)
    end_full = time.perf_counter()

    print(f'Training took {end_full-end_ab:.2f} s. \n\n Finished training...\n Process took {end_full-start_full:.2f} s.')



    return {'log_reg':log_reg_clf,
            'tree': dt_clf,
            'rf' : rf_clf,
            'adb': aboost_clf,
            'svc': svc}
    

            