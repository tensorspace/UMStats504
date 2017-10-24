import pandas as pd
import numpy as np
import statsmodels.api as sm
import patsy
from read_meps import meps

# Estimated age x sex x race population totals
ages = pd.cut(meps.AGE14X, np.arange(0, 90, 5))
asx = pd.crosstab(ages, [meps.SEX, meps.RACEV2X], values=meps.PERWT14F, aggfunc=np.sum)
print((asx / 1e6).to_string(float_format="%.2f"))

# age x sex x race cell counts from sample
asy = pd.crosstab(ages, [meps.SEX, meps.RACEV2X])

# The ratio of unweighted to weighted proportions, for sample relative
# to population.
ta = (asy / asy.sum().sum()) / (asx / asx.sum().sum())
print(ta.to_string(float_format="%.2f"))

# Estimated proportion of the population in each cell who smoke
sm1 = pd.crosstab(ages, meps.SEX, values=(meps.ADSMOK42==1)*meps.PERWT14F,
                  aggfunc=np.sum)
sm2 = pd.crosstab(ages, meps.SEX, values=(meps.ADSMOK42==2)*meps.PERWT14F,
                  aggfunc=np.sum)
smr = sm1 / (sm1 + sm2)

# Proportion of the sample in each cell who smoke
sm1x = pd.crosstab(ages, meps.SEX, values=(meps.ADSMOK42==1), aggfunc=np.sum)
sm2x = pd.crosstab(ages, meps.SEX, values=(meps.ADSMOK42==2), aggfunc=np.sum)
smrx = sm1x / (sm1x + sm2x)
