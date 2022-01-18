from google.oauth2 import service_account
from google.cloud import bigquery
import configparser
import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format


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
%matplotlib inline

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


bus_cat_holding= bus_cat_dataframe.merge(right=holding_dataframe, how='inner', on = 'BusinessName')
bus_cat_holding.shape

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