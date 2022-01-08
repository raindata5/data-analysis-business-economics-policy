from sklearn.base import BaseEstimator, TransformerMixin

class UsedCarsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, odometer_scale = False):
        self.odometer_scale = odometer_scale
    def fit(self, X, y=None):
            return self
    def transform(self, X):
        import re

        X['age'] = 2021 - X.iloc[:,name_pos].str.split(' ').str[0].astype(int)
        X['age_sq'] = X.iloc[:,-1] **2
        X['age_cu'] = X.iloc[:,-2] **3
        X['cat_xle'] = np.where(X.iloc[:,name_pos].str.contains(r'\bxle\b', flags= re.IGNORECASE), 1,0 )
        X['cat_le'] = np.where(X.iloc[:,name_pos].str.contains(r'\ble\b', flags= re.IGNORECASE),1,0 )
        X['cat_excellent'] = np.where(X.iloc[:,condition_pos] == 'excellent',1,0 )
        X['cat_good'] = np.where(X.iloc[:,condition_pos] == 'good',1,0 )
        X['cat_likenew'] = np.where(X.iloc[:,condition_pos] == 'like new',1,0 )
        X['manual'] = np.where(X.iloc[:,condition_pos] == 'manual',1,0 )
        X['other'] = np.where(X.iloc[:,condition_pos] == 'other',1,0 )
        X['odometer_flag'] = np.where(X.iloc[:,odometer_pos].isna() ,1,0 )   
        X = X.iloc[:,-12:]
        return X.values
