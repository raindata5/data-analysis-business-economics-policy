# EDA through the use of correlation and regression
from turtle import left
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
import numpy as np
#[]
#
cg_df = pd.read_parquet('cg_est', engine='pyarrow')
bh_df: pd.DataFrame = pd.read_parquet('bus_holdings', engine='pyarrow')
bus_cats_df = pd.read_parquet('bus_cats', engine='pyarrow')

#[]
# Do businesses that offer a variety (more categories) tend to have higher reviewcounts?
# we can aggregate the counts for each business
bus_cats_counts_df = bus_cats_df.groupby(['BusinessName'], as_index=False)['BusinessCategoryName'].count()
bus_cats_counts_df = bus_cats_counts_df.rename({'BusinessCategoryName': 'cat_counts'}, axis=1)
bus_cats_counts_df

#[]
# most recent business holding
most_recent_bus_df = bh_df.sort_values(by='CloseDate', ascending=True).groupby(['BusinessName'], as_index=False).last()
most_recent_bus_df

#[]
#
most_recent_bus_cats_df = pd.merge(left=most_recent_bus_df, right=bus_cats_counts_df, on='BusinessName', how='inner')
most_recent_bus_cats_df
#[]
#
for col in most_recent_bus_cats_df.filter(regex= '[Rr]ating', axis=1).columns:
    most_recent_bus_cats_df[col] = most_recent_bus_cats_df[col].astype(float)
most_recent_bus_cats_df.dtypes
#[]
# checking correlation of the business category counts and other variables
# cov[x,y] / std[x]std[y]
cat_counts_corr = most_recent_bus_cats_df.corr()['cat_counts']
cat_counts_corr.sort_values()

#[]
# 
plt.figure(figsize=(15, 10))
sns.heatmap(most_recent_bus_cats_df.corr()[:-1], xticklabels=most_recent_bus_cats_df.corr()[:-1].columns, yticklabels=most_recent_bus_cats_df.corr()[:-1].columns, cmap='coolwarm')
plt.title('Heat Map of Correlation')
plt.tight_layout()
plt.show()

#[]
#
most_recent_bus_cats_df.hist(column=[
    'cat_counts', 'total_review_cnt_delta', 'previous_review_cnt', 'ReviewCount'], figsize=(15, 10), bins=10)
#[]
# Another chart that may provide a bit more information is this regression plot with the joint distribution in the background
plt.figure(figsize=(15, 10))
sns.regplot(x=most_recent_bus_cats_df['cat_counts'], y=most_recent_bus_cats_df['total_review_cnt_delta'], x_jitter=.05)

#[]
#
def bus_cat_groups(x):
    group_dict = {}
    cats = []
    for row in x:
        cats.append(row)
    group_dict['cats'] = cats
    group_dict['goe_4'] = (1 if len(cats) >= 4 else 0)
    return pd.Series(group_dict)
bus_cats_groups_df = bus_cats_df.groupby(by=['BusinessName', 'CityName', 'CountyName', 'CountryName', 'StateName', 'Latitude','Longitude'], as_index=False, observed=True)['BusinessCategoryName'].apply(bus_cat_groups)
bus_cats_groups_df_cut = bus_cats_groups_df[bus_cats_groups_df['goe_4'] == 1]
pd.set_option('display.max_colwidth', 100) # 50 seems to be default
bus_cats_groups_df_cut

#[]
#
bus_cats_groups_df_cut_cg = pd.merge(left=bus_cats_groups_df_cut, right=cg_df, on=['StateName','CountyName'], how='left')
bus_cats_groups_df_cut_cg.head()

#[]
#
cg_df_not_in = cg_df[~(cg_df['StateName'].astype(str) + cg_df['CountyName']).isin((bus_cats_groups_df_cut_cg['StateName'].astype(str) + bus_cats_groups_df_cut_cg['CountyName']).tolist())]
cg_df_not_in.shape

#[]
#
cg_df_not_in.EstimatedPopulation.mean()
bus_cats_groups_df_cut_cg.groupby(by=['StateName','CountyName'], as_index=False, observed=True)['EstimatedPopulation'].first()

#[]
#
import plotly.graph_objects as go

fig = go.Figure(
    data=go.Scattergeo(
        lon=bus_cats_groups_df_cut_cg['Longitude'],
        lat=bus_cats_groups_df_cut_cg['Latitude'],
        mode='markers',
        text = bus_cats_groups_df_cut_cg['abs_delta'].astype(str) , 
        marker = dict(
            colorscale = 'Blues', 
            color=bus_cats_groups_df_cut_cg['abs_delta'],
            colorbar = dict(
            titleside = "top",
            outlinecolor = "rgba(68, 68, 68, 0)",
            title = "Estimated Abosute Population <br> change from previous year"

        ))
    )
)
fig.update_layout(
    geo_scope='usa',
    title= 'Businesses specializing in 4+ business category areas'
)
fig.show()

#[]
#

most_recent_bus_cats_df_lt_4 = most_recent_bus_cats_df[most_recent_bus_cats_df['cat_counts'] < 4]
most_recent_bus_cats_df_lt_4.shape

#[]
#
most_recent_bus_cats_df_not_na = most_recent_bus_cats_df[most_recent_bus_cats_df['total_review_cnt_delta'].notna()]
most_recent_bus_cats_df_not_na.info()

#[]
#
X = sm.add_constant(most_recent_bus_cats_df_not_na['cat_counts'])
y = most_recent_bus_cats_df_not_na['total_review_cnt_delta']
univariate_lin_model = sm.OLS(Y, X)
results = univariate_lin_model.fit(cov_type='HC1')
#[]
#

results.summary()
##come back and predict using the model
#[]
#
#make sure to add constant when predicting as well
univariate_lin_model_predictions = results.predict(X)
univariate_lin_model_predictions

#[]
#
univariate_lin_model_predictions_variance = ((univariate_lin_model_predictions - univariate_lin_model_predictions.mean()) ** 2).sum() / univariate_lin_model_predictions.shape[0]
y_variance = ((y - y.mean()) ** 2).sum() / y.shape[0]
univariate_lin_model_rsquared = univariate_lin_model_predictions_variance / y_variance

#[]
#
residuals = y - univariate_lin_model_predictions
residuals

#[]
#
residuals_var = ((residuals - residuals.mean()) ** 2).sum() / residuals.shape[0]
univariate_lin_model_rsquared_2 = 1 - (residuals_var / y_variance)

#[]
# 
# X_variance = np.sqrt(((X - X.mean()) ** 2).sum() / X.shape[0])

# y_X_covariance = ((y - y.mean()) * (X[1] - X[1].mean())).sum() / X.shape[0]
# slope = y_X_covariance / X_variance[1]
# slope


#[]
# on return finish decided on whether or not to remove these values
# from here do a linear regression (do with higher-order of 2?) check residuals and flag those above ordinary
# remember correlation and look up at square of CC
#[]
# seems the greatest correlation is to the total change in the review count of a business
# .12 isn't too bad either for one variable
# will take a look at some of the scatter plots

scatter_matrix(most_recent_bus_cats_df)