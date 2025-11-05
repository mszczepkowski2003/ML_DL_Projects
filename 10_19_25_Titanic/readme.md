# Titanic Survival Prediction

A classic introductory machine learning project using the famous Titanic dataset to predict which passengers survived the sinking.

## Project Goal

The primary objective of this project was to exploit and develop a better understanding of different classifcation and machine learning concepts. Through this process, I successfully implemented custom Scikit-learn Transformers, gained profound insight into utilizing Pipeline architecture for robust workflow management, and thoroughly mastered the nuances of the chosen modelling techniques.
## Technology Stack

This project is built using standard data science and machine learning libraries in Python.

* **Language:** Python 3.12
* **Core Libraries:**
    * `pandas` (Data manipulation and cleaning)
    * `numpy` (Numerical operations)
    * `scikit-learn` (Model building, training, and evaluation)
    * `matplotlib` / `seaborn` (Data visualization)

## 📁 Repository Structure:
```
.
├── data/
│   └── raw/                      # Original data files 
│
├── notebooks/                    # Jupyter Notebooks for exploration, experiments and baseline model development
│   ├── eda.ipynb                 # Exploratory Data Analysis
│   └── model_baseline_*.ipynb    # Baseline models and experimentation
│
├── src/                          # Source code for modular, production-ready components
│   ├── data/                     # Data loading utilities
│   ├── features/                 # Custom scikit-learn Transformers and Pipelines
│   ├── master/                   # Main execution script (main.py)
│   └── models/                   # Model training and testing scripts
│
├── reports/                      # Project artifacts, results, and generated outputs
│   ├── figures/                  # Generated plots and visualizations
│   └── model_outputs/            # Saved models (.pkl) and reports (Excel, JSON)
│
├── environment.yml               # Conda environment definition
└── requirements.txt              # Pip dependencies
```
## Running the project on your machine

Follow these steps to set up and run the project locally.

Download the Titanic data set from Kaggle: https://www.kaggle.com/datasets/yasserh/titanic-dataset
Save the file in the following location: data/raw/Titanic-Dataset.csv

1. Clone the repository:
    - git clone https://github.com/mszczepkowski2003/Data_science_projects.git
    - cd Data_science_projects/10_19_25_Titanic
2. Set up the environment
for conda:
    - conda env create -f environment.yml
    - conda activate titanic_env
for pip:
    - pip install -r requirements.txt
3. Run the project (full pipeline: Data loading -> preprocessing -> model training -> evaluation)
    - python -m src.master.main

If everything runs correctly the message will be shown: Pipeline process has finished with success

All trained models, model config and .xlsx report will be saved in reports/
