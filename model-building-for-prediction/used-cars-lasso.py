from pathlib import Path
import os 
import sys
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 35)
pd.set_option('display.max_rows', 200)
pd.options.display.float_format = '{:,.2f}'.format

# 1)	Clean the data a bit
# 2)	Create a complex regression model
# 3)	Use Lasso regression
# 4)	Modify the model accordingly 


path = Path(os.getcwd())
base_dir = path.parent.parent.parent
data_in = os.path.join(str(base_dir), "Desktop/Rain-data/da-for-bep/da_data_repo/used-cars/raw")

used_cars = pd.read_csv(os.path.join(data_in, "used_cars_2cities.csv"))

#[]
# exploring data a bit
used_cars.info()
used_cars.shape
used_cars.head()

#[]
# going to want to create the holdout set but going to check area and see if this warrants doing a stratified shuffle split
used_cars.area.value_counts()
#[]
#
from sklearn.model_selection import StratifiedShuffleSplit
shuffler = StratifiedShuffleSplit(n_splits=1, test_size=.20, train_size=.80, random_state=42)

for train_index, test_index in shuffler.split(used_cars, used_cars['area']):
    uc_train_set = used_cars.loc[train_index]
    uc_test_set = used_cars.loc[test_index]

#[]
#
uc_train_set.shape
uc_test_set.shape

#[]
#
uc_train_set.head()
#[]
# searching for which cat variables I want to create dummies for
uc_train_set.columns
# 'area', condition, cylinders, paintcolor, use pd qcut for odometer

['v1', 'price', 'area', 'subarea', 'name', 'condition', 
'cylinders', 'drive', 'fuel', 'odometer', 'paintcolor', 
'size', 'transmission', 'type', 'dealer']
#[]
#  for the name variable we will try and see if we can conduct any feature engineering

uc_train_set.name.str.split(' ').str[1].value_counts()
# all the cars are toyotas

# for these observations with year in the same place as the brand of the car, they can still be used since there is no problem with the year
# also we can confirm that they are all toyotas
uc_train_set.loc[uc_train_set.name.str.split(' ').str[1].str.contains(r'\d+'),'name']

#creating new column for age of vehicle
uc_train_set.name.str.split(' ').str[0].value_counts()
uc_train_set.loc[:,'age'] = 2021 - uc_train_set.name.str.split(' ').str[0].astype(int)
import re

# the model

uc_train_set.name.str.split(' ').str[2].value_counts()
# confirming that those observations that don't have the model in this positions are toyota camry's
uc_train_set.loc[uc_train_set.name.str.split(' ').str[2].str.contains(r'[^camry_]', flags= re.IGNORECASE),'name']

uc_train_set.name.str.split(' ').apply(len).value_counts()
# for the ones with this info the type e.g. xle, hybrid, etc.
uc_train_set.name.str.split(' ').str[4].value_counts()


# uc_train_set.loc[uc_train_set.name.str.contains(r'\bxle\b', flags= re.IGNORECASE),'name'].shape
from collections import Counter
Counter(np.where(uc_train_set.name.str.contains(r'\bxle\b', flags= re.IGNORECASE), 1,0 ))

uc_train_set['cat_xle'] = np.where(uc_train_set.name.str.contains(r'\bxle\b', flags= re.IGNORECASE), 1,0 )
#[]
#

uc_train_set.loc[uc_train_set.name.str.contains(r'\ble\b', flags= re.IGNORECASE),'name'].head()
# confirming that our word boundaries excluded xle
uc_train_set.loc[uc_train_set.name.str.contains(r'\ble\b', flags= re.IGNORECASE),'name'].shape
uc_train_set.loc[uc_train_set.name.str.contains(r'\ble\b', flags= re.IGNORECASE),'name'].str.contains(r'\bxle\b', flags= re.IGNORECASE)
# uc_train_set['cat_le'] = np.where(uc_train_set.name.str.contains(r'\ble\b', flags= re.IGNORECASE),1,0 )
#[]
#
uc_train_set['cat_le'] = np.where(uc_train_set.name.str.contains(r'\ble\b', flags= re.IGNORECASE),1,0 )

#[]
#
uc_train_set.condition.value_counts()
Counter(np.where(uc_train_set.condition == 'excellent',1,0 ))
uc_train_set['cat_excellent'] = np.where(uc_train_set.condition == 'excellent',1,0 )
uc_train_set['cat_good'] = np.where(uc_train_set.condition == 'good',1,0 )
uc_train_set['cat_likenew'] = np.where(uc_train_set.condition == 'like new',1,0 )


# uc_train_set.to_pickle("uc_train_set.pkl")
# uc_test_set.to_pickle("uc_test_set.pkl")
# uc_train_set = pd.read_pickle("uc_train_set.pkl")
# uc_test_set = pd.read_pickle("uc_test_set.pkl")
#[]
# dropping columns with no price
uc_train_set.dropna(subset=['price'], axis=0, inplace=True)
uc_train_set.reset_index(drop=True, inplace=True)
#[]
# generally from a statisically perspective it's better to have the category with the most values as the one "left out" in regression
uc_train_set.transmission.value_counts()
encoded_transmission = pd.get_dummies(uc_train_set.transmission, drop_first=True)

#[]
#
uc_train_set_merged = pd.merge(uc_train_set, encoded_transmission, how='inner', left_index=True, right_index=True)
#verfication
uc_train_set_merged.loc[uc_train_set_merged.transmission == 'manual', encoded_transmission.columns.tolist()]

#[]
# going to transofmr values of odometer so they're not so back, I would prefer the standardscaler from sklearn but this method
# keeps the results rather intelligible
# also adding a flag binary variable for those with missing odometer info since we will impute values
uc_train_set_merged['odometer_transformed'] = (uc_train_set_merged.odometer / 1000)
uc_train_set_merged.loc[uc_train_set_merged.odometer.isna(), ['condition']].value_counts(dropna=False)
uc_train_set_merged['missing_odometer_flag'] = np.where(uc_train_set_merged.odometer.isna(), 1, 0)
#[]
# this plot to show whether not data data is skewed as is and can help to determine whether to impute mean or median
# in future create heat map for missing values, 
uc_train_set_merged.odometer.plot(kind='hist')
plt.show()

imputer = SimpleImputer(strategy='mean', add_indicator=True)

test = imputer.fit_transform(uc_train_set_merged[['odometer_transformed']])
test_df = pd.DataFrame(data=test, columns = ['odometer_transformed', 'flag'])

#[]
#
#confirmation by comparing the flag variables and adding the true values
(pd.merge(uc_train_set_merged, test_df, how='inner', left_index=True, right_index=True)['flag'] == pd.merge(uc_train_set_merged, test_df, how='inner', left_index=True, right_index=True)['missing_odometer_flag']).sum()

uc_train_set_merged_impute = pd.merge(uc_train_set_merged, test_df, how='inner', left_index=True, right_index=True, suffixes=['','_imputer'])

# uc_train_set_merged_impute.to_pickle("test.pkl")
# uc_train_set_merged_impute = pd.read_pickle("test.pkl")

#[]
#
uc_train_set_merged_impute['price'] = uc_train_set_merged_impute.price.str.extract(r'(\d+)').astype(int)

#[]
#
sns.regplot(y='price', x='age', data=uc_train_set_merged_impute,lowess=True)
plt.show()

#[]
# going to remove these 2 outliers which are possible duplicate and also raise the price up a bit
# some values have a price of one which are rather unrealistic
uc_train_set_merged_impute.loc[uc_train_set_merged_impute['price'] > 60000]
uc_train_set_merged_impute['price'].value_counts(bins=5, normalize=True)
uc_train_set_merged_impute['price'].quantile(np.arange(0, 1.1, .1))
uc_train_set_merged_impute.loc[uc_train_set_merged_impute['price'] < 500['price']]
uc_train_set_merged_impute = uc_train_set_merged_impute.loc[uc_train_set_merged_impute['price'].between(200, 60000)]

#[]
# from lowess plot we did it seemed that we could take a highorder polynomial

# from sklearn.preprocessing import PolynomialFeatures
# poly = PolynomialFeatures(2, include_bias=False)
# poly.fit_transform([uc_train_set_merged_impute['age']]).shape
uc_train_set_merged_impute['age_sq'] = uc_train_set_merged_impute['age'] **2
uc_train_set_merged_impute['age_cu'] = uc_train_set_merged_impute['age'] **3

#[]
#
desired_columns = ['dealer','age','cat_xle','cat_le','cat_excellent','cat_good','cat_likenew','manual','other'
,'missing_odometer_flag','odometer_transformed_imputer','age_sq','age_cu']
uc_train_set_y = uc_train_set_merged_impute['price']
uc_train_set_X = uc_train_set_merged_impute[desired_columns]

#[]
#


lasso_model = Lasso(alpha=200, max_iter=2000)
lasso_model.fit(uc_train_set_X, uc_train_set_y)

#[]
#
lasso_model.intercept_
lasso_model.coef_
pd.Series(lasso_model.coef_, index = uc_train_set_X.columns)
lasso_predictions = lasso_model.predict(uc_train_set_X)
lasso_rmse = mean_squared_error(uc_train_set_y, lasso_predictions, squared=False)

#[]
#

lin_reg = LinearRegression()
lin_reg.fit(uc_train_set_X, uc_train_set_y)
lin_reg.intercept_
lin_reg.coef_
pd.Series(lin_reg.coef_, index = uc_train_set_X.columns)
lin_reg_predictions = lin_reg.predict(uc_train_set_X)
lin_reg_rmse = mean_squared_error(uc_train_set_y, lin_reg_predictions, squared=False)


#[]
#
scores = cross_val_score(lasso_model, uc_train_set_X, uc_train_set_y ,cv=4, scoring="neg_mean_squared_error")
lasso_cv_rmse = np.sqrt(-scores)
lasso_cv_rmse_mean = np.mean(lasso_cv_rmse)

#[]
#
scores = cross_val_score(lin_reg, uc_train_set_X, uc_train_set_y ,cv=4, scoring="neg_mean_squared_error")
lin_reg_cv_rmse = np.sqrt(-scores)
lin_reg_cv_rmse_mean = np.mean(lin_reg_cv_rmse)


#[]
#
uc_test_set = pd.read_pickle("uc_test_set.pkl")


uc_test_set = uc_test_set.dropna(subset=['price'], axis=0)
uc_test_set['price'] = uc_test_set.loc[:,'price'].str.extract(r'(\d+)').astype(int)
uc_test_set = uc_test_set.loc[uc_test_set.loc[:,'price'].between(200, 60000)]

uc_test_set = uc_test_set.reset_index(drop=True)

uc_test_set_y = uc_test_set['price'].copy()
uc_test_set_X = uc_test_set.drop(columns = ['price'])

uc_test_set_y.shape[0] == uc_test_set_X.shape[0]


cols = 'name', 'dealer', 'condition', 'transmission', 'odometer'
name_pos, dealer_pos, condition_pos, transmission_pos, odometer_pos  = [uc_test_set_X.columns.get_loc(col) for col in cols]
#[]]
# creating custom transformer



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

#just think of a way to remove certain columns from the pipeline as well (maybe do this based on variable)
#probably best not to return tuple since i'll be using column transformer
# create the pipeline with column transformer
trans = UsedCarsTransformer()
test = trans.transform(uc_test_set_X)

#[]
#

odometer_pipeline = Pipeline(
            [
                ('imputer', SimpleImputer(strategy='median')),
                ('std_scaler', StandardScaler())
                
                
            ]
        )
odometer_pipeline.fit_transform(uc_test_set_X['odometer'].values.reshape(-1,1))

#[]
#



full_pipeline = ColumnTransformer(
        [
            ('FeatureEngineering', UsedCarsTransformer(), list(uc_test_set_X.columns)),
            ('odometer', odometer_pipeline, ['odometer'])
        ]
    )
uc_test_set_X_tr = full_pipeline.fit_transform(uc_test_set_X)
import joblib
# joblib.dump(uc_test_set_X_tr,"uc_test_set_X_tr")
# joblib.load("uc_test_set_X_tr")
# uc_test_set_y.to_pickle("uc_test_set_y_tr.pkl")

uc_test_set_y_tr = pd.read_pickle("uc_test_set_y_tr.pkl")
uc_test_set_X_tr = joblib.load("uc_test_set_X_tr")
# uc_train_set_x = pd.read_pickle("uc_train_set_x.pkl")
# uc_train_set_y = pd.read_pickle("uc_train_set_y.pkl")

#[]
# since I decided to not transform odometer by merely dividin by 
# a thousand I will redo transformation on work set
# here just modifying path to download the files which are present in another directory
path = Path(os.getcwd())
base_dir = path.parent.parent.parent
data_in = os.path.join(str(base_dir), "Desktop/Rain-data/da-for-bep/da_data_repo/used-cars/raw")

used_cars = pd.read_csv(os.path.join(data_in, "used_cars_2cities.csv"))

#[]
# here our analysis is based on determining price so if that is na 
# then better to drop it
# also those cars under $200 tend to be unrealistic offers as well 
# as those above 60000
used_cars = used_cars.dropna(subset=['price'], axis=0)
used_cars['price'] = used_cars.loc[:,'price'].str.extract(r'(\d+)').astype(int)
used_cars = used_cars.loc[used_cars.loc[:,'price'].between(200, 60000)]
used_cars = used_cars.reset_index(drop=True)
used_cars.shape
#[]
# deciding to split by the area as this can influence the price and 
# chicago tends to have more dealers which influences the price
used_cars.groupby(['area']).mean()
used_cars.groupby(['dealer']).mean()
from sklearn.model_selection import StratifiedShuffleSplit
shuffler = StratifiedShuffleSplit(n_splits=1, test_size=.20, train_size=.80, random_state=42)

for train_index, test_index in shuffler.split(used_cars, used_cars['area']):
    uc_train_set = used_cars.loc[train_index]
    uc_test_set = used_cars.loc[test_index]

#[]
# starting with running data on work set which with cv 
# will consist of training and test set
uc_train_set_y = uc_train_set['price']
uc_train_set_X = uc_train_set.drop(columns = ['price'])

# have to change to make sure columns come from right input data
# as opposed to previous version of it
full_pipeline = ColumnTransformer(
        [
            ('FeatureEngineering', UsedCarsTransformer(), list(uc_train_set_X.columns)),
            ('odometer', odometer_pipeline, ['odometer'])
        ]
    )


uc_train_set_X_tr = full_pipeline.fit_transform(uc_train_set_X)



cols = 'name', 'dealer', 'condition', 'transmission', 'odometer'
name_pos, dealer_pos, condition_pos, transmission_pos, odometer_pos  = [uc_train_set_X.columns.get_loc(col) for col in cols]
coefficient_labels = ['dealer', 'age', 'age_sq', 'age_cu', 'cat_xle', 'cat_le',      
        'cat_excellent', 'cat_good', 'cat_likenew', 'manual', 'other', 'odometer_flag','odometer_scaled']


#[]
# training lasso model
lasso_model = Lasso(alpha=400, max_iter=2000)
lasso_model.fit(uc_train_set_X_tr, uc_train_set_y)

#[]
# kind of hard to intepret the coeffcients since the intercept starts high
# but it shows what's more or less expected 
#  although age sq coefficient is negative best ot think of this as
# a positive number considering our plot of age and price and also
# the high intercept
lasso_model.intercept_
lasso_model.coef_
pd.Series(lasso_model.coef_, index = coefficient_labels)

#[]
# the predictions are pretty bad which is why it's best to use gridsearchcv
# going to adjust lambda a bit and proceed with cv after LR
lasso_predictions = lasso_model.predict(uc_train_set_X_tr)
lasso_rmse = mean_squared_error(uc_train_set_y, lasso_predictions, squared=False)
lasso_rmse
lasso_predictions[:5]
uc_train_set_y[:5]
#[]
# here we can see the benefits of ultimately picking an unregularized
# model , it's a bit easier to draw meaning out of the coeff.
# we picked them after all
lin_reg = LinearRegression()
lin_reg.fit(uc_train_set_X_tr, uc_train_set_y)
lin_reg.intercept_
lin_reg.coef_
pd.Series(lin_reg.coef_, index = coefficient_labels)
lin_reg_predictions = lin_reg.predict(uc_train_set_X_tr)
lin_reg_rmse = mean_squared_error(uc_train_set_y, lin_reg_predictions, squared=False)
lin_reg_rmse # far better rmse

#[]
#
from sklearn.model_selection import GridSearchCV
param_grid = {'alpha': np.arange(0,1001,50), 'positive': [True, False]}
gs_lasso = GridSearchCV(lasso_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)

#[]
# Lasso(alpha=0, max_iter=2000) is the best so I will
# try one more time with smaller lambda values
gs_lasso.fit(uc_train_set_X_tr, uc_train_set_y)
gs_lasso.best_params_
gs_lasso.best_estimator_

#[]
# so alpha=5 is better
param_grid2 = {'alpha': np.arange(0,51,5), 'positive': [True, False]}
gs_lasso2 = GridSearchCV(lasso_model, param_grid=param_grid2, scoring='neg_mean_squared_error', cv=5)
gs_lasso2.fit(uc_train_set_X_tr, uc_train_set_y)
gs_lasso2.best_params_
gs_lasso2.best_estimator_
cv_results = gs_lasso2.cv_results_

import joblib

uc_train_set_X.to_pickle("uc_train_set_X.pkl")
uc_train_set_y.to_pickle("uc_train_set_y.pkl")
uc_test_set.to_pickle("uc_test_set.pkl")

gs_lasso2 = joblib.load("gs_lasso2")
full_pipeline = joblib.load("full_pipeline")
uc_train_set_X = pd.read_pickle("uc_train_set_X.pkl")
uc_train_set_y = pd.read_pickle("uc_train_set_y.pkl")
uc_test_set = pd.read_pickle("uc_test_set.pkl")

#[]
# 2 methods for recording the results in a csv
# but the scores are interesting since they don't change too much for the first couple of models
results = []
for p,mts in zip(cv_results['params'], cv_results['mean_test_score']):
    model = []
    model.append(p)
    model.append(np.sqrt(-mts))
    results.append(model)

with open('filename2.csv','w') as f:
    writer = csv.writer(f, delimiter= '|')
    writer.writerows(results)

f.close()

results2 = sorted([[np.sqrt(-mts),p] for p,mts in zip(cv_results['params'], cv_results['mean_test_score'])], reverse=False)

with open('gs_lasso2_cv_results.csv','w') as f:
    writer = csv.writer(f, delimiter= '|')
    writer.writerows(results2)

f.close()

#[]
# important variables to look at are age_sq, and odometer flag
# odometer hints that perhaps those that don't place the odometer information are hiding something,
# either way they have lower prices on their ads
cols = 'name', 'dealer', 'condition', 'transmission', 'odometer'
name_pos, dealer_pos, condition_pos, transmission_pos, odometer_pos  = [uc_train_set_X.columns.get_loc(col) for col in cols]
coefficient_labels = ['dealer', 'age', 'age_sq', 'age_cu', 'cat_xle', 'cat_le',      
        'cat_excellent', 'cat_good', 'cat_likenew', 'manual', 'other', 'odometer_flag','odometer_scaled']

uc_train_set_X_tr = full_pipeline.fit_transform(uc_train_set_X)
lin_reg = LinearRegression()
lin_reg.fit(uc_train_set_X_tr, uc_train_set_y)
lin_reg.intercept_
lin_reg.coef_
pd.Series(lin_reg.coef_, index = coefficient_labels)


lin_reg_coeff = sorted(zip(lin_reg.coef_,coefficient_labels ), reverse=False) #variable coefficients
with open('lin_reg_variable_coeff.csv','w') as f:
    writer = csv.writer(f, delimiter= '|')
    writer.writerows(lin_reg_coeff)

f.close()


#[]
# expected thise LR model to get a worse score than the lasso since I have a cubed variable
# which was implemented to try to cause overfitting
# perhaps more interactions and higher order variables would be needed since technically my model isn't that complex
scores = cross_val_score(lin_reg, uc_train_set_X_tr, uc_train_set_y ,cv=5, scoring="neg_mean_squared_error")
lin_reg_cv_rmse = np.sqrt(-scores)
lin_reg_cv_rmse_mean = np.mean(lin_reg_cv_rmse) #2391.23 as opposed to 2392 on best model for lasso


#[]
#
uc_test_set_X = uc_test_set.drop(columns='price')
uc_test_set_y = uc_test_set['price']

uc_test_set_X_tr = full_pipeline.fit_transform(uc_test_set_X)

predictions_lin_reg_final = lin_reg.predict(uc_test_set_X_tr)
rmse_lin_reg_final = mean_squared_error(uc_test_set_y, predictions_lin_reg_final, squared=False)
rmse_lin_reg_final # 2352 


#[]
# checking lasso can see it still brings down age cubed
gs_lasso_be_coeff = sorted(zip(gs_lasso2.best_estimator_.coef_, coefficient_labels))

with open('gs_lasso_be_coeff.csv','w') as f:
    writer = csv.writer(f, delimiter= '|')
    writer.writerows(gs_lasso_be_coeff)

f.close()

final_gs_lasso_model = gs_lasso2.best_estimator_


#[]
# so the regular lin reg would've done better after all
final_gs_lasso_model = gs_lasso2.best_estimator_
predictions_final_gs_lasso_model = final_gs_lasso_model.predict(uc_test_set_X_tr)
rmse_final_gs_lasso_model = mean_squared_error(uc_test_set_y, predictions_final_gs_lasso_model, squared=False)
rmse_final_gs_lasso_model # 2354

#[]
# 95% confidence interval , where we can expect the rmse to be 

from scipy import stats

confidence = 0.95
squared_errors = (predictions_lin_reg_final - uc_test_set_y) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                         loc=squared_errors.mean(),
                         scale=stats.sem(squared_errors)))
# [1840.68777491, 2771.39439297]

#[]
# going to preserve the model, it's predictions, transformation pipeline, and would do trained parameters but there was no need
joblib.dump(lin_reg, "lin_reg_model")
joblib.dump(full_pipeline, "uc_transformation_pipeline")
joblib.dump(predictions_lin_reg_final, "lin_reg_model_predictions")


# 
#
#[]


import csv




    
f.close()




# also test full pipeline on base data after requirements (drop nulls,etc.)
#[]
#



#[]
#



# transformations
#     age column
#     cat_xle , le
#     cat excellent, good, likenew
#     dropping rows with null price
#     season dummies on transmission (in a pipeline I wouldn't want to use pd get_dummis)
#     transformed odometer by dividing by 1000 
#     imputed odometer median to values with null and added a binary variable to each #
#     square and cube on price
#     changed price to integer

# i can come back and find more features
# uc_train_set.loc[(uc_train_set['cat_xle'] == 0) & (uc_train_set['cat_le'] == 0),['cat_le','cat_xle']].value_counts()
# uc_train_set['missing_cat_type'] = np.where((uc_train_set['cat_xle'] == 0) & (uc_train_set['cat_le'] == 0), 1, 0)

# ['dealer', 'age', 'age_sq', 'age_cu', 'cat_xle', 'cat_le',      
#        'cat_excellent', 'cat_good', 'cat_likenew', 'manual', 'other', odometer_flag,odometer_scaled]