import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(path):
    return pd.read_csv(path)

def tr_te_split(raw_df, target='Survived', test_size=0.2):
    
    X = raw_df.drop(target, axis=1)
    y = raw_df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y,
                                                        test_size=test_size,
                                                        random_state=42)
    return X_train, X_test, y_train, y_test