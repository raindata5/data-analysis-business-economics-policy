from google.oauth2 import service_account
from google.cloud import bigquery
import configparser
import pandas as pd
import numpy as np
import math
import seaborn as sns
import plotly.express as px 
import matplotlib.pyplot as plt
%matplotlib inline
pd.options.display.float_format = '{:,.2f}'.format
pd.options.display.float_format = '{:,.4f}'.format

KEY_PATH = "/mnt/c/Users/Ron/git-repos/yelp-data/gourmanddwh-f75384f95e86.json"
CREDS = service_account.Credentials.from_service_account_file(KEY_PATH)
client = bigquery.Client(credentials=CREDS, project=CREDS.project_id)

#[]
#
cg_file = open('county_growth_est.sql','r')
county_growth_query =  cg_file.read()

cg_dataframe = (
    client.query(county_growth_query)
    .result()
    .to_dataframe()
)

cg_dataframe.to_csv('county_growth_est.csv',sep='|', float_format = '{:,.2f}', index=False )

#[]
#
holding_file = open('business_daily_holding.sql')
holding_query = holding_file.read()

holding_dataframe = (
    client.query(holding_query)
    .result()
    .to_dataframe()
)


#[]
#
bus_cat_file = open('business_category_location.sql')
bus_cat_query = bus_cat_file.read()

bus_cat_dataframe = (
    client.query(bus_cat_query)
    .result()
    .to_dataframe()
)

#[]
#
cg_dataframe

#[]
#
holding_dataframe

#[]
#
holding_dataframe.info()

#[]
#
holding_dataframe['CloseDate'] = pd.to_datetime(holding_dataframe['CloseDate'])
holding_dataframe[['BusinessRating', 'previous_rating', 'abs_rating_diff', 'total_bus_rating_delta']] = holding_dataframe[['BusinessRating', 'previous_rating', 'abs_rating_diff', 'total_bus_rating_delta']].astype(float)

#[]
#
holding_dataframe.info()

#[]
#
holding_dataframe

#[]
#
bus_cat_dataframe

#[]
bus_cat_dataframe.info()

#[]
#
cg_dataframe.to_csv('county_growth_est.csv',sep='|', index=False )
holding_dataframe.to_csv('holding.csv',sep='|', index=False )
bus_cat_dataframe.to_csv('bus_cat.csv',sep='|', index=False )

#[]
#
cg_dataframe = pd.read_csv('county_growth_est.csv',sep='|', low_memory=True)
holding_dataframe = pd.read_csv('holding.csv',sep='|', low_memory=True)
bus_cat_dataframe = pd.read_csv('bus_cat.csv',sep='|', low_memory=True)

#[]
#
cg_dataframe

#[]
holding_dataframe
holding_dataframe.sort_values(['total_review_cnt_delta', 'CloseDate'], ascending = False).head(10)

#[]
#
most_recent_holding = holding_dataframe.sort_values(['BusinessName', 'CloseDate'], ascending = False).drop_duplicates(subset=['BusinessName'] , keep='first')
print(most_recent_holding.shape)
print(most_recent_holding)

#[]
#
bus_cat_dataframe
#[]
#
import matplotlib.pyplot as plt


#[]
#
most_recent_holding = most_recent_holding.reset_index(drop=True)
most_recent_holding

#[]
#
most_recent_holding.hist(bins=20,figsize=(15,20))

#[]
#
print(most_recent_holding['total_bus_rating_delta'].value_counts(bins=10, normalize=True, sort=False))
print('\n')
print(most_recent_holding['total_bus_rating_delta'].value_counts(bins=10, sort=False))

#[]
#
thirdq, firstq = most_recent_holding['total_review_cnt_delta'].quantile(.75), most_recent_holding['total_review_cnt_delta'].quantile(.25)
quartilerange = 1.5 * (thirdq - firstq)
highoutlier, lowoutlier = quartilerange + thirdq, firstq - quartilerange
print(highoutlier, lowoutlier, sep='<---->')

#[]
#
thirdq2, firstq2 = most_recent_holding['total_bus_rating_delta'].quantile(.75), most_recent_holding['total_bus_rating_delta'].quantile(.25)
quartilerange2 = 1.5 * (thirdq2 - firstq2)
highoutlier2, lowoutlier2 = quartilerange2 + thirdq2, firstq2 - quartilerange2
print(highoutlier2, lowoutlier2, sep='<---->')

#[]
#
def get_outliers():
  dfout = pd.DataFrame(columns = most_recent_holding.columns, data=None) #initializes a dataframe with no values but all orig. columns from df
  for col in ['total_bus_rating_delta', 'total_review_cnt_delta']: # just going to loop through the numeric columns
    thirdq, firstq = most_recent_holding[col].quantile(0.75), most_recent_holding[col].quantile(0.25)
    quartilerange = 1.5*(thirdq-firstq)
    highoutlier, lowoutlier = quartilerange + thirdq, firstq - quartilerange
    df = most_recent_holding.loc[(most_recent_holding[col] > highoutlier) | (most_recent_holding[col] < lowoutlier)] # for each columns we will isolate the extreme values
    df = df.assign(varname = col, threshlow= lowoutlier, threshhigh= highoutlier) # creates 3 new columns that corresponds to a label and the high outlier , and then the low outlier for that label respectively
    dfout = pd.concat([dfout,df]) # just a simple concatenation of the df's
  return dfout

extreme_df = get_outliers()

#[]
#
grouped_extreme_df = extreme_df.groupby(['BusinessName'], as_index=False)['varname'].count()
grouped_extreme_df_bus = grouped_extreme_df.loc[grouped_extreme_df['varname'] > 1, ['BusinessName']]
grouped_extreme_df_bus

#[]
#
exe_most_recent_holding = most_recent_holding.loc[most_recent_holding['BusinessName'].isin(grouped_extreme_df_bus['BusinessName'].values), :]
exe_most_recent_holding = exe_most_recent_holding.reset_index(drop=True)
exe_most_recent_holding

#[]
#

print(exe_most_recent_holding['total_bus_rating_delta'].value_counts( ascending=False).sort_index())
print('\n')
print(exe_most_recent_holding['total_review_cnt_delta'].value_counts(ascending=False).sort_index())

#[]
#
exe_most_recent_holding.describe(include='all')

#[]
#
exe_most_recent_holding.to_csv('exe_most_recent_holding.csv',sep='|', index=False )

#[]
#
exe_most_recent_holding_loc = bus_cat_dataframe.merge(right=exe_most_recent_holding, how='inner', on = 'BusinessName')
exe_most_recent_holding_loc

#[]
#
df3 = pd.merge(left=df2, right = cg_dataframe, left_on=['CountyName','StateName'], right_on=['CountyName','StateName'])
df3

#[]
#
df3.columns

#[]
#
bins_of_10 = pd.cut(df3['abs_review_diff'], bins=10)
bins_of_3 = pd.cut(df3['abs_review_diff'], bins=3)

#[]
#
df_bins = pd.concat([bins_of_3, bins_of_10], axis=1)
df_bins.columns = ['bins_of_3', 'bins_of_10']

df3_w_bins =  pd.concat([df3, df_bins], axis=1)
df3_w_bins

#[]
#
df3_w_bins.groupby(['bins_of_3'], as_index=False).agg({'bins_of_3': ['median']})


bus_cat_holding= bus_cat_dataframe.merge(right=holding_dataframe, how='inner', on = 'BusinessName')
bus_cat_holding.shape

#[]
#
fig = px.scatter(
  df=df3_w_bins,
  x='total_review_cnt_delta',
  y='abs_delta'
)

fig.show()

#[]
#


#[]
# first we'll group by categoryname and see the agg results
cat_groups = bus_cat_holding.groupby(['BusinessCategoryName'], as_index=False)[['ReviewCount','BusinessRating']].agg({"ReviewCount": ['sum', 'mean', 'max'], "BusinessRating": ['mean', 'max']})

#[]
#



# here we'll focus on the top 10 based on the following criteria
cat_groups.sort_values(by=[('ReviewCount', 'sum'), ('ReviewCount', 'mean'), ('BusinessRating', 'mean')], ascending=False).head(10)
#[]
#
bus_cat_holding.groupby(['BusinessCategoryName'], as_index=False)['BusinessName'].count().sort_values(by=[
    'BusinessName'], ascending=False).head(10)

cat_groups.reset_index()

#[]
#
cg_dataframe = pd.read_csv('county_growth_est.csv',sep='|', low_memory=True)
holding_dataframe = pd.read_csv('holding.csv',sep='|', low_memory=True)
bus_cat_dataframe = pd.read_csv('bus_cat.csv',sep='|', low_memory=True)
pd.options.display.float_format = '{:,.4f}'.format

#[]
#

#variance

est_pop_variance = ((cg_dataframe.EstimatedPopulation - cg_dataframe.EstimatedPopulation.mean()) ** 2).sum() / cg_dataframe.EstimatedPopulation.shape[0]
est_pop_variance

# standard deviation
est_pop_std = (((cg_dataframe.EstimatedPopulation - cg_dataframe.EstimatedPopulation.mean()) ** 2).sum() / cg_dataframe.EstimatedPopulation.shape[0]) ** (1/2)
est_pop_std

std2 = np.std(cg_dataframe.EstimatedPopulation)
std2
#variance
abs_delta_variance =  ((cg_dataframe.abs_delta - cg_dataframe.abs_delta.mean()) ** 2).sum() / cg_dataframe.abs_delta.shape[0]
abs_delta_variance
# standard deviation
abs_delta_std = abs_delta_variance ** (1/2)
abs_delta_std

# Covariance

covariance_abs_delta_est_pop = ((cg_dataframe.abs_delta - cg_dataframe.abs_delta.mean()) * (cg_dataframe.EstimatedPopulation - cg_dataframe.EstimatedPopulation.mean())).sum() / cg_dataframe.EstimatedPopulation.shape[0]
covariance_abs_delta_est_pop

# correlation of abs_Delta and est_pop
corr_abs_delta_est_pop = covariance_abs_delta_est_pop / (est_pop_std * abs_delta_std)
corr_abs_delta_est_pop

cg_dataframe.corr()['abs_delta']

fig = px.scatter(cg_dataframe, x='EstimatedPopulation', y='abs_delta')
fig.show()

#[]
# take z scores to see which states are the "biggest" (latent variable) with the z-scores coming from proxy variables
    # take the avg horizontally
# check and see how correlations can differ between different states  
    # use a for loop to through each state with boolean indexing

cg_dataframe['abs_delta_z_score'] = (cg_dataframe.abs_delta - cg_dataframe.abs_delta.mean()) / abs_delta_std

cg_dataframe['est_pop_z_score'] = (cg_dataframe.EstimatedPopulation - cg_dataframe.EstimatedPopulation.mean()) / est_pop_std

cg_dataframe['avg_z_score'] = cg_dataframe[['est_pop_z_score', 'abs_delta_z_score']].mean(axis=1)

cg_dataframe.sort_values(['avg_z_score'], ascending=False)

state_abs_est_corrs = []
for state in cg_dataframe.StateName.unique():
  state_filtered_df = cg_dataframe.loc[cg_dataframe.StateName == state]
  cov_abs_est = ((state_filtered_df['abs_delta'] - state_filtered_df['abs_delta'].mean()) * (state_filtered_df['EstimatedPopulation'] - state_filtered_df['EstimatedPopulation'].mean())).sum() / state_filtered_df.shape[0]
  state_filtered_df_abs_delta_std = (((state_filtered_df['abs_delta'] - state_filtered_df['abs_delta'].mean()) ** 2).sum() / state_filtered_df.shape[0]) ** (1/2)
  state_filtered_df_est_pop_std = (((state_filtered_df['EstimatedPopulation'] - state_filtered_df['EstimatedPopulation'].mean()) ** 2).sum() / state_filtered_df.shape[0]) ** (1/2)
  state_abs_est_corr = cov_abs_est / (state_filtered_df_abs_delta_std * state_filtered_df_est_pop_std)
  state_abs_est_corr_tuple = (state, state_abs_est_corr, state_filtered_df.shape[0])
  state_abs_est_corrs.append(state_abs_est_corr_tuple)

state_abs_est_corrs_df = pd.DataFrame(data = state_abs_est_corrs, columns=['statename', 'abs_delta_estimated_pop_corr', 'obs_cnt'])

state_abs_est_corrs_df.sort_values('abs_delta_estimated_pop_corr', ascending=False)

# district of columnbia seems to be Nan because there is only one observation
runtime_warning = cg_dataframe.loc[cg_dataframe.StateName == 'District of Columbia'].corr()['abs_delta']



# sampledata = pd.Series([170,155,160,185,145])

# std = math.sqrt((((sampledata - sampledata.mean()) ** 2).sum() / sampledata.shape[0]))
# std

# std = (((sampledata - sampledata.mean()) ** 2).sum() / sampledata.shape[0]) **.5
# std