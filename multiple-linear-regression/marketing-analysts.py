#[]
#
import pandas as pd
import numpy as np

cps2014 = pd.read_pickle('views/cps2014_analysis_analyst2.pkl')

cps2014.info()
cps2014.head()

#[]
#
cps2014 = cps2014.iloc[:,2:]
cps2014.info()
cps2014.head()

#[]
#
import matplotlib.pyplot as plt
cps2014.plot(kind='scatter', x='age', y='ln_wage')
plt.show()



#[]
# simple linear piecewise regression
import statsmodels.formula.api as smf
from patsy import dmatrices
from py_helper_functions import *

model = smf.ols("ln_wage ~ lspline(age,65)", data=cps2014).fit(cov_type='HC1')
model.summary()

y_hat = model.predict(cps2014.age.sort_values())

#trying to plot the piecewise linear regression but this isn't the best example especially considering the r-squared value
plt.scatter(cps2014['age'], cps2014['ln_wage'])
plt.scatter(cps2014['age'],  y_hat, color='red' )
plt.show()


#[]
# multiple regression with piecewise linear spline

import statsmodels.api as sm
cps2014.uhourse = pd.to_numeric(cps2014.uhourse)
cps2014.uhourse.value_counts()

y1, X1 = dmatrices("ln_wage ~ lspline(uhourse,40) + female", cps2014)
lps_model = sm.OLS(y1, X1).fit()

lps_model.summary()

cps2014['ln_wage_hat'] = lps_model.predict(X1)
fig, ax = plt.subplots(ncols=1)
ax.scatter(x=cps2014['ln_wage_hat'],y=cps2014.ln_wage, color='red',alpha=0.4)
ax.set_xlabel('predicted ln_wage')
ax.set_ylabel('ln_wage')
ax.axline([0, 0], [1, 1])
plt.show()


