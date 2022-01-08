import os
import sys
from pathlib import Path
from numpy.core.fromnumeric import mean
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
#[]
# chaning path to read in data
path = Path(os.getcwd())
base_dir = path.parent.parent.parent
data_dir = os.path.join(str(base_dir), "Desktop/Rain-data/da-for-bep/da_data_repo/used-cars/raw/")

#[]
# making sure there are no unexpected changes in the data
used_cars = pd.read_csv(os.path.join(data_dir, "used_cars_2cities.csv"))
used_cars.info()
used_cars.shape

#[]
# getting rid of nulls first, also excluding some observations that don't interest us, and separating independent an dependent variables
used_cars = used_cars.dropna(subset=['price'], axis=0)
used_cars['price'] = used_cars.loc[:,'price'].str.extract(r'(\d+)').astype(int)
used_cars = used_cars.loc[used_cars.loc[:,'price'].between(200, 60000)]
used_cars = used_cars.reset_index(drop=True)

#[]
#
from sklearn.model_selection import StratifiedShuffleSplit
shuffler = StratifiedShuffleSplit(n_splits=1, test_size=.20, train_size=.80, random_state=42)

for train_index, test_index in shuffler.split(used_cars, used_cars['area']):
    uc_train_set = used_cars.loc[train_index]
    uc_test_set = used_cars.loc[test_index]

#[]
#
uc_train_set_X = uc_train_set.drop(columns='price')
uc_train_set_y = uc_train_set['price']

#[]
#



#[]
# apart from importing the transformation pipeline to use it, I have to import a couple of modules and retrieve some classes

# can't import the UsedCarsTransformer because there are dashes in the title of it's main file and parent directory

cols = 'name', 'dealer', 'condition', 'transmission', 'odometer'
name_pos, dealer_pos, condition_pos, transmission_pos, odometer_pos  = [uc_train_set_X.columns.get_loc(col) for col in cols]
coefficient_labels = ['dealer', 'age', 'age_sq', 'age_cu', 'cat_xle', 'cat_le',      
        'cat_excellent', 'cat_good', 'cat_likenew', 'manual', 'other', 'odometer_flag','odometer_scaled']




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



trans_pipeline = joblib.load(os.path.join(str(base_dir), "git-repos/data-analysis-business-economics-policy/model-building-for-prediction/uc_transformation_pipeline"))


#[]
# 
uc_train_set_X_tr = trans_pipeline.fit_transform(uc_train_set_X)

#[]
# here we are going to show how a regular decision tree algorithm without any stopping rules or pruning will overfit the data set


tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(uc_train_set_X_tr, uc_train_set_y)
tree_reg_predictions = tree_reg.predict(uc_train_set_X_tr)

tree_reg_rmse = mean_squared_error(uc_train_set_y, tree_reg_predictions, squared=False)
tree_reg_rmse # at 544 rmse it doesn't seem so bad

#[]
# Now we'll undergo cross validation to help confirm whether this is a case of overfitting or not 

scores = cross_val_score(tree_reg, uc_train_set_X_tr, uc_train_set_y, scoring="neg_mean_squared_error")
tree_reg_cv_rmse_mean = np.mean(np.sqrt(-scores))
tree_reg_cv_rmse_mean # cv paints a different picture with the rmse now being 1629

#[]
# a complexity parameter wouldn't solve every problem with this approach but it will help prevent overfitting of the data
tree_reg_cp = DecisionTreeRegressor(random_state=42, min_samples_leaf=5)
tree_reg_cp.fit(uc_train_set_X_tr, uc_train_set_y)
tree_reg_cp_predictions = tree_reg_cp.predict(uc_train_set_X_tr)

tree_reg_cp_rmse = mean_squared_error(uc_train_set_y, tree_reg_cp_predictions, squared=False)
tree_reg_cp_rmse # at 1424 rmse

#[]
# doesn't seem that the model was overfitting as much as thought, it may be due to the number of samples
cp_scores = cross_val_score(tree_reg_cp, uc_train_set_X_tr, uc_train_set_y, scoring="neg_mean_squared_error")
tree_reg_cp_cv_rmse_mean = np.mean(np.sqrt(-cp_scores))
tree_reg_cp_cv_rmse_mean # 1824 when cross validated

#[]
# I'm going to build a learning curve and see what happens there
from sklearn.model_selection import train_test_split
# uc_train_set_X
# uc_train_set_y
train_X, val_X, train_y, val_y = train_test_split(uc_train_set_X, uc_train_set_y, test_size=.20, random_state=42)

train_X_tr = trans_pipeline.fit_transform(train_X)
val_X_tr = trans_pipeline.fit_transform(val_X)



#[]
# based on this one would say that the model overfits the train data due to the gap between train and val
# also when the model overfits the more data you feed it the more the RMSE would decrease
import matplotlib.pyplot as plt

rmses_train, rmses_val = [], []

for row in range(1, len(train_X) + 1) :
    tree_reg.fit(train_X_tr[:row], train_y[:row])
    predictions_train = tree_reg.predict(train_X_tr[:row])
    rmse_train = mean_squared_error(train_y[:row], predictions_train, squared=False)
    rmses_train.append(rmse_train)

    predictions_val = tree_reg.predict(val_X_tr)
    rmse_val = mean_squared_error(val_y, predictions_val, squared=False)
    rmses_val.append(rmse_val)

plt.plot(rmses_train, "r-+", linewidth=2, label="train")
plt.plot(rmses_val, "b-", linewidth=3, label="val")
plt.legend(loc="upper right", fontsize=14)   
plt.xlabel("Training set sample size", fontsize=12)
plt.ylabel("RMSE", fontsize=14)  
plt.show()

#[]
# we'll make this into a function this time to facilitate recalling it
# here we can see a difference . There are two indications that this model doesn't overfit the model
# one is how close the validation and training set RMSE is to each other
# also now if the model underfits there would be a plateau with the RMSE where it wouldn't decrease any further
# this may not be the exact case here but it applies more so then with our other model
# to improve we would have to increase the complexity of our model or do more feature engineering


def chart_learning_curve(model, X_train, val_X_train, y_train, val_y):

    rmses_train, rmses_val = [], []

    for row in range(1, len(X_train) + 1) :
        model.fit(X_train[:row], y_train[:row])
        predictions_train = model.predict(X_train[:row])
        rmse_train = mean_squared_error(y_train[:row], predictions_train, squared=False)
        rmses_train.append(rmse_train)

        predictions_val = model.predict(val_X_train)
        rmse_val = mean_squared_error(val_y, predictions_val, squared=False)
        rmses_val.append(rmse_val)

    plt.plot(rmses_train, "r-+", linewidth=2, label="train")
    plt.plot(rmses_val, "b-", linewidth=3, label="val")
    plt.legend(loc="upper right", fontsize=14)   
    plt.xlabel("Training set sample size", fontsize=12)
    plt.ylabel("RMSE", fontsize=14)  
    plt.show()

chart_learning_curve(tree_reg_cp, train_X_tr, val_X_tr, train_y, val_y)

#[]
#  one important thing to mention is how much outliers can affect a regression tree
# Why? due to the cutoff values and the splits that occur

used_cars = pd.read_csv(os.path.join(data_dir, "used_cars_2cities.csv"))
used_cars = used_cars.dropna(subset=['price'], axis=0)
used_cars['price'] = used_cars.loc[:,'price'].str.extract(r'(\d+)').astype(int)

used_cars_outliers = used_cars.reset_index(drop=True)



uc_train_set_outliers_X = used_cars_outliers.drop(columns='price')
uc_val_set_outliers_y = used_cars_outliers['price']

#[]
# This is a good example of the problem
train_X_outliers, val_X_outliers, train_y_outliers, val_y_outliers = train_test_split(uc_train_set_outliers_X, uc_val_set_outliers_y, test_size=.20, random_state=42)

uc_train_set_outliers_X_tr = trans_pipeline.fit_transform(train_X_outliers)
val_X_outliers_tr = trans_pipeline.fit_transform(val_X_outliers)

chart_learning_curve(tree_reg_cp, uc_train_set_outliers_X_tr, val_X_outliers_tr, train_y_outliers, val_y_outliers)

#[]
# before was 1824
# using np.append to put the validation set and training set back together so that cross_val_score can split them instead

cp_scores_outliers = cross_val_score(tree_reg_cp, np.append(uc_train_set_outliers_X_tr,val_X_outliers_tr, axis=0), np.append(train_y_outliers,val_y_outliers, axis=0), scoring="neg_mean_squared_error")
tree_reg_cp_cv_rmse_mean_outliers = np.mean(np.sqrt(-cp_scores_outliers))
tree_reg_cp_cv_rmse_mean_outliers # now 4098 rmse



