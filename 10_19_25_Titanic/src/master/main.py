from src.data.load_data import load_dataset, tr_te_split
from src.visualization.eda_plot import make_plots
from src.models.train import train_all_models
from src.models.test import test_all_models, create_report
import numpy as np


SEED = 42
np.random.seed(SEED)
def main():
# load data
    raw_df = load_dataset('data/raw/Titanic-Dataset.csv')
# split the data
    X_train, X_test, y_train, y_test = tr_te_split(raw_df)
# EDA plots on Train
    
# EDA Tables -- skip for now
# Train models

    trained_models = train_all_models(X_train, y_train, SEED)
# Test models that takes trained models dictionary and evaluates the predictions
    evaluated_models = test_all_models(trained_models, X_test)
# Create report in .xlsx format
    create_report(evaluated_models, X_train, y_train, y_test, trained_models)

    print("Pipeline process has finished with success")

if __name__ == '__main__':
    main()