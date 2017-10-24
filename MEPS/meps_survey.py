# To do survey analysis in Python we need to use a non-standard branch
# from github.
import sys
sys.path.insert(0, "/afs/umich.edu/user/k/s/kshedden/jarvis/statsmodels")

import pandas as pd
import numpy as np
import statsmodels.api as sm
import patsy
from read_meps import meps

ages = pd.cut(meps.AGE14X, np.arange(0, 90, 5))

# Total office-visit expenses in billions, by age/sex category
di = sm.survey.SurveyDesign(strata=meps.VARSTR, cluster=meps.VARPSU,
                            weights=meps.PERWT14F, nest=True)
sj, se = [], []
for sex in 1,2:
    for age in ages.cat.categories:
        x = ((meps.SEX == sex) & (ages == age)) * meps.OBDEXP14
        svm = sm.survey.SurveyTotal(di, x)
        sj.append([sex, age, svm.est[0]])
        se.append([sex, age, svm.stderr[0]])
sj = pd.DataFrame(sj)
sj.columns = ["Sex", "Age", "Prop"]
sj = sj.pivot(index="Age", columns="Sex", values="Prop")
sj /= 1e9
se = pd.DataFrame(se)
se.columns = ["Sex", "Age", "SE"]
se = se.pivot(index="Age", columns="Sex", values="SE")
se /= 1e9
print(sj.to_string(float_format="%.2f"))
print(se.to_string(float_format="%.2f"))

# Total office-visit expenses in billions, by age/sex category, using
# fake clusters to illustrate the impact of clustering on the standard
# errors.
di = sm.survey.SurveyDesign(strata=meps.VARSTR, cluster=np.random.choice(100, meps.shape[0]),
                            weights=meps.PERWT14F, nest=True)
sj2, se2 = [], []
for sex in 1,2:
    for age in ages.cat.categories:
        x = ((meps.SEX == sex) & (ages == age)) * meps.OBDEXP14
        svm = sm.survey.SurveyTotal(di, x)
        sj2.append([sex, age, svm.est[0]])
        se2.append([sex, age, svm.stderr[0]])
sj2 = pd.DataFrame(sj2)
sj2.columns = ["Sex", "Age", "Prop"]
sj2 = sj2.pivot(index="Age", columns="Sex", values="Prop")
sj2 /= 1e9
se2 = pd.DataFrame(se2)
se2.columns = ["Sex", "Age", "SE"]
se2 = se2.pivot(index="Age", columns="Sex", values="SE")
se2 /= 1e9
print(sj2.to_string(float_format="%.2f"))
print(se2.to_string(float_format="%.2f"))


meps["agecats"] = ages
mx = meps[["INSCOV14", "SEX", "agecats", "PERWT14F", "VARPSU", "VARSTR"]].dropna()
di = sm.survey.SurveyDesign(strata=mx.VARSTR, cluster=mx.VARPSU,
                            weights=mx.PERWT14F, nest=True)
y,x = patsy.dmatrices("INSCOV14 ~ SEX + agecats", data=mx)
y = 1*(y == 3)
model = sm.survey.SurveyModel(di, sm.GLM, init_args={"family": sm.families.Binomial()})
result = model.fit(y, x, cov_method="jackknife")
rslt = pd.DataFrame({"Names": x.design_info.column_names, "Params": model.params,
                     "SE": model.stderr})
