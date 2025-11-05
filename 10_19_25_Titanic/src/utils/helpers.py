import pandas as pd 
import numpy as np
import re
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, classification_report
from sklearn.model_selection import RandomizedSearchCV
import seaborn as sns 
import matplotlib.pyplot as plt 
def one_ticket(ticket_id, ticket_counts):
    """Helper function for transforming the n of people on the same ticket"""
    return 1 if ticket_counts[ticket_id] > 1 else 0

def add_ticket_repetition(df) -> pd.DataFrame:
    """
    Adds a column 'ticket_repeated', that points if a ticket is used
    by more than one person 
    0 - one person for one ticket
    1 - more than one person for one ticket
    """
    grp_tickets = df.groupby('Ticket')['PassengerId'].count()
    df['Ticket_repeated'] = df['Ticket'].apply(lambda x: one_ticket(x, grp_tickets))
    return df.drop('Ticket',axis=1)

def other_titles(X_train) -> list:
    titles2oth = X_train['Personal_title'].unique()
    x = lambda x: x if x in titles2oth else 'Other'
    return x

def get_personal_title(df) -> pd.DataFrame:
    regex = r', (\w+)\.'
    df['Personal_title'] = df['Name'].apply(lambda x: (re.search(regex, x).group(1)) if (re.search(regex, x)) else None)
    title_counts = df['Personal_title'].value_counts()
    df.loc[df['Personal_title'].isna(),'Personal_title'] = 'Mrs'
    df['Personal_title'] = df['Personal_title'].apply(lambda x: 'Other' if title_counts[x]<= 40 else x)
    return df

def gender_map(df):
    sex_map = {'female' : 0,
               'male' : 1}

    df['Sex'] = df['Sex'].map(sex_map)
    return df


def grid_search_res(gs_fitted, res_df=None):
    cv_results = gs_fitted.cv_results_
    config = str(gs_fitted.best_params_)
    best_index = gs_fitted.best_index_
    mean_test_accuracy = cv_results['mean_test_accuracy'][best_index]
    mean_test_precision = cv_results['mean_test_precision'][best_index]
    mean_test_recall = cv_results['mean_test_recall'][best_index]
    mean_test_f1 = cv_results['mean_test_f1'][best_index]
    mean_train_f1 = cv_results['mean_train_f1'][best_index]
    new_df = pd.Series(data={'f1' : mean_test_f1,
                                'Accuracy' : mean_test_accuracy,
                                'Precision' : mean_test_precision,
                                    'Recall': mean_test_recall,
                                    'mean_train_f1' : mean_train_f1,
                                    'P_grid_config': config}).to_frame()
    if res_df is None:
        return new_df
        
    else:
        df = pd.concat([res_df, new_df], axis = 1)

    print("Cross-validation performance (best model):")
    print(f"Accuracy: {mean_test_accuracy:.3f}")
    print(f"Precision (weighted): {mean_test_precision:.3f}")
    print(f"Recall (weighted): {mean_test_recall:.3f}")
    print(f"F1 (weighted): {mean_test_f1:.3f}")
    print(f"F1 (weighted) OVERFIT CHECK: {mean_train_f1:.3f}")
    return df


def plot_confusion_matrix(y_true, y_pred):
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred) #2
    fig = disp.figure_
    plt.tight_layout()
    return fig

def plot_roc(y_true, y_pred):
    disp = RocCurveDisplay.from_predictions(y_true, y_pred)  
    fig = disp.figure_
    plt.tight_layout()
    return fig

def classification_report_df(y_true, y_pred):
    x = classification_report(y_true, y_pred, output_dict=True)
    return round(pd.DataFrame(x).T,3)


def random_search_cv(pipe, param_dist, refit='f1',n_iter=50): 
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision_weighted',
        'recall': 'recall_weighted',
        'f1': 'f1_weighted',
    }
    rs = RandomizedSearchCV(pipe, 
                    param_distributions=param_dist, 
                    scoring=scoring, 
                    refit=refit,
                    n_jobs=-1,
                    return_train_score=True,
                        verbose=1,
                        n_iter = n_iter,
                        )
    return rs

