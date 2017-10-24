import patsy
import pandas as pd
import numpy as np
from drug_categories import dcat
import statsmodels.api as sm
from sklearn import linear_model
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

fname = "PartD_Prescriber_PUF_NPI_Drug_15.txt.gz"

dfx = pd.read_csv(fname, delimiter="\t")

# Create a variable indicating which drugs are opioids
dfx["opioid"] = dfx.drug_name.isin(dcat["opioid"])

# Get the total day supply for each provider, for opioids and non-opioids
dr = dfx.groupby(["npi", "opioid"]).agg({"total_day_supply": np.sum})
dr = dr.unstack()
dr = dr.rename(columns={False: "Non_opioids", True: "Opioids"})
dr = dr.fillna(0)
dr = dr.reset_index()
dr.columns = ["npi", "Non_opioids", "Opioids"]

# Merge in the state where the provider operates
ds = dfx.groupby("npi").agg({'nppes_provider_state': "first", "nppes_provider_city": "first"})
ds = ds.reset_index()
dr = pd.merge(dr, ds, left_on="npi", right_on="npi")
dr = dr.rename(columns={"nppes_provider_state": "state", "nppes_provider_city": "city"})

dr = dr.loc[dr.Opioids > 0, :]
dr = dr.loc[dr.Non_opioids > 0, :]

# Remove very small states
s = dr.groupby("state").size()
s = pd.DataFrame({"num_providers": s})
dr = pd.merge(dr, s, left_on="state", right_index=True, how='left')
dr = dr.loc[dr.num_providers >= 500]

# Create log transformed variables and z-scores
dr["log_op"] = np.log2(0.1 + dr.Opioids)
dr["log_nonop"] = np.log2(0.1 + dr.Non_opioids)
dr["log_op_z"] = (dr.log_op - dr.log_op.mean()) / dr.log_op.std()
dr["log_nonop_z"] = (dr.log_nonop - dr.log_nonop.mean()) / dr.log_nonop.std()

dr = dr.dropna()

pdf = PdfPages("opioids.pdf")

# Plot log/log data
plt.clf()
plt.plot(dr.log_op, dr.log_nonop, 'o', color='orange', alpha=0.3, rasterized=True)
plt.xlabel("Non-opioid days supply")
plt.ylabel("Opioids days supply")
plt.grid(True)
pdf.savefig()

# Basic log/log regression model
olsmodel = sm.OLS.from_formula("log_op ~ log_nonop", data=dr)
olsresult = olsmodel.fit()
olsmodelz = sm.OLS.from_formula("log_op_z ~ log_nonop_z", data=dr)
olsresultz = olsmodelz.fit()

# Fit by state
plt.clf()
xr = np.linspace(-4, 4, 20)
state_params = []
for dy in dr.groupby("state"):
    sn = dy[0] # State abbreviation
    dz = dy[1] # Data
    model = sm.OLS.from_formula("log_op_z ~ log_nonop_z", data=dz)
    result = model.fit()
    params = result.params.values
    state_params.append(params)
    plt.plot(xr, params[0] + params[1]*xr, '-', color='grey')
plt.title("Trends by state")
plt.xlabel("Log non-opioid days supply (Z-score)")
plt.ylabel("Log opioid days supply (Z-score)")
pdf.savefig()

state_params = np.asarray(state_params)

pdf.close()

# Fixed effects for states
femodel = sm.OLS.from_formula("log_op ~ log_nonop + C(state)", data=dr)
feresult = femodel.fit()
femodelz = sm.OLS.from_formula("log_op_z ~ log_nonop_z + C(state)", data=dr)
feresultz = femodelz.fit()
femodelsz = sm.OLS.from_formula("log_op_z ~ log_nonop_z * C(state)", data=dr)
feresultsz = femodelsz.fit()

# Compare state intercepts
pa = feresultz.params
vc = feresultz.cov_params()
ii = [i for i,x in enumerate(pa.index) if x.startswith("C(state")]
pa = pa.iloc[ii].values
vc = vc.iloc[ii, ii].values
zc = np.zeros((len(pa), len(pa)))
for i1 in range(len(pa)):
    for i2 in range(len(pa)):
        d = np.zeros(len(pa))
        d[i1] = 1
        d[i2] = -1
        se = np.sqrt(np.dot(d, np.dot(vc, d)))
        zc[i1, i2] = (pa[i1] - pa[i2]) / se
j1, j2 = np.tril_indices(len(pa), -1)
zc = zc[j1, j2]

# Compare state intercepts
pa = feresultsz.params
vc = feresultsz.cov_params()
ii = [i for i,x in enumerate(pa.index) if x.startswith("log_nonop_z:C(state")]
pa = pa.iloc[ii].values
vc = vc.iloc[ii, ii].values
zcx = np.zeros((len(pa), len(pa)))
for i1 in range(len(pa)):
    for i2 in range(len(pa)):
        d = np.zeros(len(pa))
        d[i1] = 1
        d[i2] = -1
        se = np.sqrt(np.dot(d, np.dot(vc, d)))
        zcx[i1, i2] = (pa[i1] - pa[i2]) / se
j1, j2 = np.tril_indices(len(pa), -1)
zcx = zcx[j1, j2]


# Mean/variance analysis
dr["fit"] = olsresult.fittedvalues
dr["bin"] = pd.qcut(dr.fit, 20)
mv = dr.groupby("bin")["fit", "log_op"].agg({"fit": np.mean, "log_op": np.var})

# Basic mixed model (random intercepts by state)
memodel = sm.MixedLM.from_formula("log_op_z ~ log_nonop_z", groups="state", data=dr)
meresult = memodel.fit()

# Allow slopes to vary by state
memodel2 = sm.MixedLM.from_formula("log_op_z ~ log_nonop_z", groups="state",
                                   vc_formula={"s": "0+log_nonop:C(state)"},
                                   data=dr)
meresult2 = memodel2.fit()

# Basic mixed model for cities (random intercepts by city)
#memodelc = sm.MixedLM.from_formula("log_op_z ~ log_nonop_z", groups="city", data=dr)
#meresultc = memodelc.fit()

db = pd.read_csv("2015_utilization_reduced.csv.gz")

# Merge provider type
du = db.groupby("npi")["provider_type"].agg("first")
du = pd.DataFrame(du).reset_index()
dr = pd.merge(dr, du, left_on="npi", right_on="npi")

# Use lars to consider provider effects
y, x = patsy.dmatrices("log_op_z ~ 0 + log_nonop_z + C(provider_type)", data=dr,
                       return_type='dataframe')
xa = np.asarray(x)
ya = np.asarray(y)[:,0]
xnames = x.columns.tolist()
alphas, active, coefs = linear_model.lars_path(xa, ya, method='lars', verbose=True)

# Display the first few variables selected by lars and the fitted
# correlation.
for k in range(1, 20):
    print(xnames[active[k-1]])
    f = np.dot(xa, coefs[:, k])
    print(np.corrcoef(ya, f)[0, 1])

# Fixed effects model for provider type
pfemodelz = sm.OLS.from_formula("log_op_z ~ log_nonop_z + C(provider_type)",
                                data=dr)
pferesultz = pfemodelz.fit()

# Basic mixed model (random intercepts by provider)
pmemodelz = sm.MixedLM.from_formula("log_op_z ~ log_nonop_z", groups="provider_type", data=dr)
pmeresultz = pmemodelz.fit()

# Merge the blups with the fixed effects estimates
re = pmeresultz.random_effects
re = {k: re[k].groups for k in re.keys()}
re = pd.DataFrame({"re": pd.Series(re)})
fe = {}
for k in pferesultz.params.index:
    if k != "Intercept" and k != "log_nonop_z":
        v = k.replace("C(provider_type)[T.", "")
        v = v[0:-1]
        fe[v] = pferesultz.params[k]
fe = pd.Series(fe)
fe = pd.DataFrame(fe, columns=["fe"])
refe = pd.merge(fe, re, left_index=True, right_index=True)
refe -= refe.mean(0)
n = dr.provider_type.value_counts()
n = pd.DataFrame(n)
n.columns=["n"]
refe = pd.merge(refe, n, left_index=True, right_index=True)

#
pt = dr.provider_type.value_counts()
ii = pt.index[pt < 100]
refe.loc[ii, :].dropna()

pmemodels = sm.MixedLM.from_formula("log_op_z ~ log_nonop_z",
                                    groups="provider_type",
                                    vc_formula={"i": "0+C(provider_type)",
                                                "s": "0+log_nonop_z:C(provider_type)"},
                                    data=dr)
pmeresults = pmemodels.fit()

# Use lars to consider provider effects
y, x = patsy.dmatrices("log_op_z ~ 0 + log_nonop_z + C(provider_type)", data=dr,
                       return_type='dataframe')
xa = np.asarray(x)
ya = np.asarray(y)[:,0]
xnames = x.columns.tolist()
alphas, active, coefs = linear_model.lars_path(xa, ya, method='lars', verbose=True)

# Display the first few variables selected by lars and the fitted
# correlation.
for k in range(1, 20):
    print(xnames[active[k-1]])
    f = np.dot(xa, coefs[:, k])
    print(np.corrcoef(ya, f)[0, 1])

pa = pd.Series(coefs[:, 15], index=xnames)
pa = pa[pa != 0]

# Use lars to consider provider effects and state effects together
y, x = patsy.dmatrices("log_op_z ~ 0 + log_nonop_z + C(provider_type) + C(state)", data=dr,
                       return_type='dataframe')
xa = np.asarray(x)
ya = np.asarray(y)[:,0]
xnames = x.columns.tolist()
alphas, active, coefs = linear_model.lars_path(xa, ya, method='lars', verbose=True)

# Display the first few variables selected by lars and the fitted
# correlation.
for k in range(1, 20):
    print(xnames[active[k-1]])
    f = np.dot(xa, coefs[:, k])
    print(k, np.corrcoef(ya, f)[0, 1])

pa = pd.Series(coefs[:, 15], index=xnames)
pa = pa[pa != 0]

# Merge procedure types
du = db.groupby(["npi", "hcpcs_code"])["line_srvc_cnt"].agg(np.sum)
du = du.unstack(fill_value=0)

from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=50, n_iter=7, random_state=42)
xr = svd.fit_transform(du.values)
xr = pd.DataFrame(xr, index=du.index)
xr.columns = ["proc_%02d" % k for k in range(xr.shape[1])]
for c in dr.columns:
    if c.startswith("proc"):
        dr = dr.drop(c, axis=1)
dr = pd.merge(dr, xr, left_on="npi", right_index=True)
ps = " + ".join(xr.columns.tolist())

y, x = patsy.dmatrices("log_op_z ~ 0 + log_nonop_z + %s" % ps,
                       data=dr, return_type='dataframe')
xa = np.asarray(x)
ya = np.asarray(y)[:,0]
xnames = x.columns.tolist()
alphas, active, coefs = linear_model.lars_path(xa, ya, method='lars', verbose=True)

# Display the first few variables selected by lars and the fitted
# correlation.
for k in range(1, 60):
    print(xnames[active[k-1]])
    f = np.dot(xa, coefs[:, k])
    print(k, np.corrcoef(ya, f)[0, 1])
