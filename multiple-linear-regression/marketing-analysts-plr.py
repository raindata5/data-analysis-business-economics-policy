#[1]
# reading in the data and inspecting data structure
import pandas as pd
import numpy as np

cps2014 = pd.read_pickle('views/cps2014_analysis_analyst2.pkl')
cps2014.info()
cps2014.head()

#[2]
# getting rid of some columns that weren't needed but also could have use pd.drop
cps2014 = cps2014.iloc[:,2:]
cps2014.info()
cps2014.head()

#[3]
# looking at the correlation between age and wage
import matplotlib.pyplot as plt
cps2014.plot(kind='scatter', x='age', y='ln_wage')
plt.show()



#[4]
# simple linear piecewise regression
import statsmodels.formula.api as smf
from py_helper_functions import *
# the following models do not take into account using a test set to see whether the model generalizes or not, however overfitting of the data is kept in mind
sm_plr_model = smf.ols("ln_wage ~ lspline(age,65)", data=cps2014).fit(cov_type='HC1')
sm_plr_model.summary()

y_hat = sm_plr_model.predict(cps2014.age)

#trying to plot the piecewise linear regression but this isn't the best example especially considering the r-squared value
# however it does show good evidence that at age 65 and above there is an associated drop in wage considering the confidence interval, p-value, and even the t-statistic
plt.scatter(cps2014['age'], cps2014['ln_wage'])
plt.scatter(cps2014['age'],  y_hat, color='red' )
plt.show()


#[]
# multiple regression with piecewise linear spline
from patsy import dmatrices
import statsmodels.api as sm
cps2014.uhourse = pd.to_numeric(cps2014.uhourse)
cps2014.uhourse.value_counts()

y1, X1 = dmatrices("ln_wage ~ lspline(uhourse,40) + female", cps2014) #creating 2 knots in the pls which are separated at 40
mul_pls_model = sm.OLS(y1, X1).fit(cov_type='HC1')

lps_model.summary()

cps2014['ln_wage_hat'] = lps_model.predict(X1)
fig, ax = plt.subplots(ncols=1)
ax.scatter(x=cps2014['ln_wage_hat'],y=cps2014.ln_wage, color='red',alpha=0.4)
ax.set_xlabel('predicted ln_wage')
ax.set_ylabel('ln_wage')
ax.axline([0, 0], [1, 1])
plt.show()

#[]
#

y2, X2= dmatrices("ln_wage ~ lspline(age,65) + female",cps2014)

mul_pls_model_2 = sm.OLS(y2, X2).fit(cov_type='HC1')
mul_pls_model_2.summary()

cps2014['ln_wage_hat_2'] = mul_pls_model_2.predict(X2)
fig, ax = plt.subplots(1,1)
ax.scatter(x=cps2014['ln_wage_hat_2'], y=cps2014.ln_wage,color='red',alpha=0.4 )
ax.set_xlabel('predicted ln_wage')
ax.set_ylabel('ln_wage')
ax.axline([0, 0], [1, 1])
plt.show()
