# do 2 regressions (normal linear, and poly., ) (the last one with poly. should do)
#     visualize the CIs of the predicted values (both regressions)
            #maybe pred. int.
#     compare through hypothesis test and see if the same or not (for now CI) (perhaps consider bootstrap) (251)
#     choose the best one and compare external validity across (space:state, time: during holidays and after, subgroup: different businesses)
# present the regression results (stargazer)

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
sys.path.append('..')
from packages import utils
#[]
#
cg_df = pd.read_parquet('cg_est', engine='pyarrow')
bh_df: pd.DataFrame = pd.read_parquet('bus_holdings.snappy.parquet', engine='pyarrow')
bus_cats_df = pd.read_parquet('bus_cats', engine='pyarrow')
bt_df = pd.read_parquet('business_transactions.snappy.parquet', engine='pyarrow')

#[]
#
most_recent_bh_tran_df = utils.most_recent_businesses_df(df1 = bh_df, df2= bt_df, on= 'BusinessKey', remove_na=True)

#[]
#
most_recent_bh_tran_sin_cero_df = most_recent_bh_tran_df[most_recent_bh_tran_df['ReviewCount'] > 0 ]
most_recent_bh_tran_sin_cero_df.loc[:, 'ln_review_count'] = np.log(most_recent_bh_tran_sin_cero_df['ReviewCount'])
sns.set_theme()
plt.figure(figsize=(15, 10))
sns.regplot(
    x=most_recent_bh_tran_sin_cero_df['BusinessRating'],
    y=most_recent_bh_tran_sin_cero_df['ln_review_count'], x_jitter=.30, order=2,
    line_kws={'color':'cyan'})

plt.title('higher-order polynomial of explanatory variable')

#[]
#
most_recent_bh_tran_sin_cero_df.loc[:, 'BusinessRating_squared'] = most_recent_bh_tran_sin_cero_df.loc[:, 'BusinessRating'] ** 2
X_br_brsquared = most_recent_bh_tran_sin_cero_df.loc[:, ['BusinessRating_squared', 'BusinessRating']]
X_br_brsquared_with_constant = sm.add_constant(X_br_brsquared)
y = most_recent_bh_tran_sin_cero_df['ln_review_count']
univariate_lin_model_rc_br = sm.OLS(y, X_br_brsquared_with_constant)
univariate_lin_model_rc_br_results = univariate_lin_model_rc_br.fit(cov_type='HC1')
univariate_lin_model_rc_br_results.summary()

#[]
#

sns.set_theme()
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
sns.regplot(
    x=most_recent_bh_tran_sin_cero_df['BusinessRating'],
    y=most_recent_bh_tran_sin_cero_df['ln_review_count'], order=2,scatter=False, ci=99, n_boot=1000,
    line_kws={'color':'cyan'}, label= "Quadratic", ax=ax)

sns.regplot(
    x=most_recent_bh_tran_sin_cero_df['BusinessRating'],
    y=most_recent_bh_tran_sin_cero_df['ln_review_count'], scatter=False, ci=99, n_boot=1000,
    line_kws={'color':'red', 'linestyle': ':'}, label="w/o additional trans.",  ax=ax)

sns.regplot(
    x=most_recent_bh_tran_sin_cero_df['BusinessRating'],
    y=most_recent_bh_tran_sin_cero_df['ln_review_count'], scatter=False,
    line_kws={'color':'green', 'linestyle': '--'}, lowess=True, label="lowess",  ax=ax)
plt.legend()
plt.title('CI\'s of various plots')

#[]
#
from stargazer.stargazer import Stargazer

#[]
#
univariate_lin_model_rc_br_results.conf_int()

#[]
#
import random
random.seed(42)
coeff_results = []
for i in range(500):

    ixs = random.choices(population = most_recent_bh_tran_sin_cero_df.index.tolist() ,k=most_recent_bh_tran_sin_cero_df.shape[0])
    sample_df = most_recent_bh_tran_sin_cero_df.loc[ixs,:]

    X_br_brsquared_sample = sample_df.loc[:, ['BusinessRating_squared', 'BusinessRating']]
    X_br_brsquared_sample_with_constant = sm.add_constant(X_br_brsquared_sample)

    y_sample = sample_df['ln_review_count']
    univariate_lin_model_rc_br_sample = sm.OLS(y_sample, X_br_brsquared_sample_with_constant)
    univariate_lin_model_rc_br_sample_results = univariate_lin_model_rc_br_sample.fit(cov_type='HC1')

    br2 = univariate_lin_model_rc_br_sample_results.params[1]
    br = univariate_lin_model_rc_br_sample_results.params[-1]
    coeff_result = br - br2
    coeff_results.append(coeff_result)

#[]
#
the_stat = univariate_lin_model_rc_br_results.params[-1] - univariate_lin_model_rc_br_results.params[1]

the_stat
#[]
#
coeff_results_se = np.std(coeff_results)
coeff_results_se

#[]
#
the_stat / coeff_results_se

#[]
#
print(f'Difference of two slope coeffcients CI with 95% confidence {the_stat - (coeff_results_se * 2):.5f} <---> {the_stat + (coeff_results_se * 2):.5f} ', end=' ')

#[]
#
