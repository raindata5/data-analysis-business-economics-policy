# 1)	Clean the data a bit
# 2)	Create a complex regression model
# 3)	Use Lasso regression
# 4)	Modify the model accordingly 

from pathlib import Path
import os 
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyreadstat

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 35)
pd.set_option('display.max_rows', 200)
pd.options.display.float_format = '{:,.2f}'.format

path = Path(os.getcwd())
base_dir = path.parent.parent.parent
data_in = os.path.join(str(base_dir), "Desktop/Rain-data/da-for-bep/da_data_repo/airbnb/raw/")
airbnb = pd.read_csv(os.path.join(data_in, "listings.csv"))
airbnb.info()

#[]
#
airbnb_london = pd.read_csv(os.path.join(data_in, "airbnb_london_listing.csv"))