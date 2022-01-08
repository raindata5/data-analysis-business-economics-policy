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
# serializing before running any transormations and also to get proper pandas object with hints 
cg_dataframe.to_csv('county_growth_est.csv',sep='|', float_format = '{:,.2f}', index=False )
holding_dataframe.to_csv('holding.csv',sep='|', float_format = '{:,.2f}', index=False )
bus_cat_dataframe.to_csv('bus_cat.csv',sep='|', float_format = '{:,.2f}', index=False )

#[]
cg_dataframe = pd.read_csv('county_growth_est.csv',sep='|', low_memory=True)
holding_dataframe = pd.read_csv('holding.csv',sep='|', low_memory=True)
bus_cat_dataframe = pd.read_csv('bus_cat.csv',sep='|', low_memory=True)

#[]
#
bus_cat_holding= bus_cat_dataframe.merge(right=holding_dataframe, how='inner', on = 'BusinessName')
bus_cat_holding.shape

#[]
# first we'll group by categoryname and see the agg results
cat_groups = bus_cat_holding.groupby(['BusinessCategoryName'], as_index=False)[['ReviewCount','BusinessRating']].agg({"ReviewCount": ['sum', 'mean', 'max'], "BusinessRating": ['mean', 'max']})

# here we'll focus on the top 10 based on the following criteria
cat_groups.sort_values(by=[('ReviewCount', 'sum'), ('ReviewCount', 'mean'), ('BusinessRating', 'mean')], ascending=False).head(10)


cat_groups.reset_index()