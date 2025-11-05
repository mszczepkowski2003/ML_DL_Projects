from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.utils.validation import check_is_fitted
import pandas as pd 
import re

class PersonalTitleTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, treshold=40, default_title_fem='Mrs', default_title_mal='Mr'):
        self.treshold = treshold
        self.default_title_fem = default_title_fem
        self.default_title_mal = default_title_mal
        self.other_titles_ = None
    def fit(self, X, y=None):
        if 'Name' not in X.columns:
            raise ValueError('X must contain Personal title column')
        regex = r', (\w+)\.'
        X['Personal_title'] = X['Name'].apply(lambda x: re.search(regex, x).group(1) if re.search(regex, x) else None) 
        X.loc[(X['Personal_title'].isna()) & (X['Sex'] == 'female'), 'Personal_title'] = self.default_title_fem 
        X.loc[(X['Personal_title'].isna()) & (X['Sex'] == 'male'), 'Personal_title'] = self.default_title_mal
        titles = X['Personal_title']
        title_counts = titles.value_counts()
        self.other_titles_ = title_counts[title_counts <= self.treshold].index.tolist()
        self.known_titles_ = set(titles.unique())
        return self
    
    def transform(self, X):
        check_is_fitted(self, 'other_titles_')
        check_is_fitted(self, 'known_titles_')

        X = X.copy()
        regex =  r', (\w+)\.'
        X['Personal_title'] = X['Name'].apply(lambda x: re.search(regex, x).group(1) if re.search(regex, x) else None)
        X.loc[(X['Personal_title'].isna()) & (X['Sex'] == 'female'), 'Personal_title'] = self.default_title_fem 
        X.loc[(X['Personal_title'].isna()) & (X['Sex'] == 'male'), 'Personal_title'] = self.default_title_mal
        X['Personal_title'] = X['Personal_title'].apply(lambda x: 'Other' if (x in self.other_titles_) or (x not in self.known_titles_) else x)
        return X.copy().drop(['Name'], axis=1)

class ColumnDropTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, columns_):
        self.columns_ = columns_
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        return X.drop(self.columns_, axis=1)
    
class ColumnNameRestorer(TransformerMixin, BaseEstimator):
    def __init__(self, feature_names):
        self.feature_names = feature_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return pd.DataFrame(X, columns=self.feature_names)
    

class ImputerDf(TransformerMixin, BaseEstimator):
    def __init__(self, imputer=None):
        self.imputer = imputer or IterativeImputer(imputation_order='ascending',
                                        random_state = 42,
                                        min_value=1,
                                        max_value=80,
                                        tol = 0.00001)
    def fit(self, X, y=None):
        self.imputer.fit(X)
        self.columns_ = X.columns
        return self
    def transform(self, X):
        return pd.DataFrame(data = self.imputer.transform(X), 
                            columns=self.columns_)
    
class NamesWrapper(TransformerMixin, BaseEstimator):
    def __init__(self, transformer=None):
        self.transformer = transformer
    def fit(self, X, y=None):
        self.transformer.fit(X)
        self.columns_ = X.columns
        return self
    def transform(self, X):
        return pd.DataFrame(data = self.transformer.transform(X), 
                            columns=self.columns_)


class NamesRestorer(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        old_cols = X.columns
        self.new_cols_ = []
        regex = r'__(\w+)'
        for col in old_cols:
            self.new_cols_.append(re.search(regex, col).group(1))
        return self
    def transform(self, X):
        X = X.copy()
        X.columns = self.new_cols_
        return X