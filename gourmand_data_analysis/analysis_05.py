# EDA through the use of correlation and regression
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
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

bus_cats_groups_df = bus_cats_df.groupby('BusinessName', as_index=False)['BusinessCategoryName'].apply(bus_cat_groups)
bus_cats_groups_df_cut = bus_cats_groups_df[bus_cats_groups_df['goe_4'] == 1]

#[]
#
pd.set_option('display.max_colwidth', 100) # 50 seems to be default
bus_cats_groups_df_cut.iloc[:,:2]

#[]
# on return finish decided on whether or not to remove these values
# from here do a linear regression (do with higher-order of 2?) check residuals and flag those above ordinary
# remember correlation and look up at square of CC
#[]
# seems the greatest correlation is to the total change in the review count of a business
# .12 isn't too bad either for one variable
# will take a look at some of the scatter plots

scatter_matrix(most_recent_bus_cats_df)