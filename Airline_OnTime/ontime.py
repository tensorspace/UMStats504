import pandas as pd
import numpy as np
import statsmodels.api as sm

# Download data from
# https://www.transtats.bts.gov/tables.asp?DB_ID=120
#
# Click the "download" link, then click the "pre-zipped file" button
# and all the fields will be selected. Then download the zip file,
# unzip it, and recompress with gzip.
df = pd.read_csv("On_Time_On_Time_Performance_2017_1.csv.gz")

#
# Scaling and mean/variance relationships among flight duration and distance
#

# Extract two key quantitative variables
dd = df[["Distance", "AirTime"]].dropna()

# Simple correlation coefficient
print(np.corrcoef(dd.T))

# Regress duration on distance
m1 = sm.OLS.from_formula("AirTime ~ Distance", dd).fit()
print(m1.summary())

# log/log regression of duration on distance
m2 = sm.OLS.from_formula("I(np.log2(AirTime)) ~ I(np.log2(Distance))", dd).fit()
print(m2.summary())

# Square root scale regression of duration on distance
m3 = sm.OLS.from_formula("I(np.sqrt(AirTime)) ~ I(np.sqrt(Distance))", dd).fit()
print(m3.summary())

# Fit a regression to assess for a linear mean/variance relationship
dd["absresid"] = np.abs(m1.resid)
m4 = sm.OLS.from_formula("absresid ~ Distance", dd).fit()
print(m4.summary())

# Check the mean/variance relationship in more detail by stratifying
print(dd.groupby(pd.qcut(dd.Distance, 10))["absresid"].aggregate(np.mean))

# Check the mean/variance relationship for the log/log regression
dd["absresidlog"] = np.abs(m2.resid)
print(dd.groupby(pd.qcut(dd.Distance, 10))["absresidlog"].aggregate(np.mean))

# Check the mean/variance relationship for the sqrt/sqrt regression
dd["absresidsqrt"] = np.abs(m3.resid)
print(dd.groupby(pd.qcut(dd.Distance, 10))["absresidsqrt"].aggregate(np.mean))

#
# Variance decomposition
#

dd = df[["OriginAirportID", "DestAirportID", "ArrDelay"]].dropna()

gb = dd.groupby(["OriginAirportID", "DestAirportID"])["ArrDelay"]

# Conditional mean and variance by origin x destination pair.
cm = gb.mean().values
cv = gb.var(ddof=0).values

# Weights (relative sample size
n = gb.size().values
n = n.astype(np.float64)
w = n / n.sum()

# Mean of the conditional variances
mcv = np.dot(w, cv)

# Variance of the conditional means
vcm = np.dot(w, (cm - dd.ArrDelay.mean())**2)

# Marginal variance (to check)
mv = dd.ArrDelay.var(ddof=0)
assert(np.abs(mv - (mcv + vcm)) < 1e-8)

# Create a map from airport ID to the 3 letter airport abbreviation
dm = pd.crosstab(df.Origin, df.OriginAirportID)
dmap = {}
for i in range(dm.shape[0]):
    j = np.argmax(dm.iloc[i,:].values)
    dmap[dm.columns[j]] = dm.index[i]

# Get the origin/destination cross-tab, then reorder so that rows and
# columns are in 1-1 correspondence.
dc = pd.crosstab(df.OriginAirportID, df.DestAirportID)
ii = set(dc.index.tolist()) & set(dc.columns)
ii = list(ii)
ii.sort()
dc = dc.loc[ii, ii]

# Compare weighted and unweighted node degree
x1 = (dc > 0).sum(1).values
x2 =  dc.sum(1).values
y1 = np.log(x1)
y2 =  np.log(x2)
z1 = (y1 - y1.mean()) / y1.std()
z2 = (y2 - y2.mean()) / y2.std()
ii = np.argmax(np.abs(z1 - z2))

# Use SVD to factorize the table of origin/destination data
u, s, vt = np.linalg.svd(dc, 0)
v = vt.T

# Try to make the SV's as positive as possible
for k in range(5):
    if (u[:, k] > 0).sum() < (u[:, k] < 0).sum():
        u[:, k] *= -1
        v[:, k] *= -1
na = dc.columns.tolist()

for k in range(5):
    print((u[:, k] * v[:, k] >= 0).mean())

sv0 = pd.DataFrame({"origin": u[:, 0], "destination": v[:, 0]}, index=[dmap[x] for x in na])
sv0 = sv0.sort_values(by="origin")

sv1 = pd.DataFrame({"origin": u[:, 1], "destination": v[:, 1]}, index=[dmap[x] for x in na])
sv1 = sv1.sort_values(by="origin")

sv2 = pd.DataFrame({"origin": u[:, 2], "destination": v[:, 2]}, index=[dmap[x] for x in na])
sv2 = sv2.sort_values(by="origin")

#
# Aircraft-focused analyses
#

du = pd.crosstab(df.TailNum, df.OriginAirportID)
da = (du > 0).sum(1)

dx = df.sort_values(by=["TailNum", "FlightDate"])

gb = dx.groupby("TailNum")
pr = []
for g in gb:
    x = g[1]["OriginAirportID"].values
    pr.append(np.mean(x[2:] == x[0:-2]))
pr = np.asarray(pr)
pr = pr[np.isfinite(pr)]

#
# Markov chain analysis
#

dp = dc.div(dc.sum(1), axis=0)
s, u = np.linalg.eig(dp.T)
assert(np.max(np.abs(s[0] - 1)) < 1e-10)
u = u[:,0].real
if np.sum(u < 0) > np.sum(u > 0):
    u *= -1
assert(np.max(np.abs(np.dot(dp.T, u) / u - 1)) < 1e-10)
u /= u.sum()
assert(np.max(np.abs(np.dot(u, dp) - u)) < 1e-10)


dx = df.groupby(["UniqueCarrier", "TailNum"])["AirTime"].agg([np.sum, np.size])
dx = dx.reset_index()
dx = dx.groupby("UniqueCarrier").agg({"sum": np.mean, "size": np.sum})
dx["sum"] /= 60
dx = dx.rename(columns={"sum": "Hours", "size": "Flights"})
dx = dx.sort_values(by="Hours")
