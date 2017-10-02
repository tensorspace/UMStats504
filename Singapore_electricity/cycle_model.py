import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.dates as mdates

pdf = PdfPages("sg_electricity.pdf")

df = pd.read_csv("sg_electricity.csv")
df["date"] = pd.to_datetime(df.date)

# Clean up the hour variable and create a unified date variable
def f(x):
    u = x.split(":")
    return int(u[0]) + int(u[1]) / 60 - 0.5

df["hourofday"] = df.period_ending_time.apply(f)
df["date"] += pd.to_timedelta(df.hourofday, 'h')


# Below are functions to extract different elements of the date
# variable, in most cases using sin/cos to force continuous
# periodicity.

# Trend (non-periodic)
def q0(x):
    return (x - pd.to_datetime("2012-01-01")).dt.days

# Periodic cycle by year
def f1(x):
    return np.cos(2 * np.pi * x.dt.dayofyear / 365)
def f2(x):
    return np.sin(2 * np.pi * x.dt.dayofyear / 365)

# Periodic cycle by week
def g1(x):
    return np.cos(2 * np.pi * (24*x.dt.dayofweek + x.dt.hour + x.dt.minute/60) / (7*24))
def g2(x):
    return np.sin(2 * np.pi * (24*x.dt.dayofweek + x.dt.hour + x.dt.minute/60) / (7*24))

# Periodic cycle by day
def h1(x):
    return np.cos(2 * np.pi * (x.dt.hour + x.dt.minute/60) / 24)
def h2(x):
    return np.sin(2 * np.pi * (x.dt.hour + x.dt.minute/60) / 24)

# Illustrative plot about spline bases
x = np.linspace(0, 2*np.pi, 100)
dd = pd.DataFrame({"x": x, "c": np.cos(x), "s": np.sin(x), "y": np.zeros(len(x))})

# B-spline basis functions
model = sm.OLS.from_formula("y ~ 0 + bs(x, df=5)", dd)
plt.clf()
plt.title("B-spline basis functions")
for j in range(model.exog.shape[1]):
    plt.plot(model.exog[:, j], '-')
pdf.savefig()

# B-spline basis functions applied to cosine
model = sm.OLS.from_formula("y ~ bs(c, df=5)", dd)
plt.clf()
plt.title("B-splines composed with cosine functions")
for j in range(model.exog.shape[1]):
    plt.plot(model.exog[:, j], '-')
pdf.savefig()

# B-spline basis functions applied to sine
model = sm.OLS.from_formula("y ~ bs(s, df=5)", dd)
plt.clf()
plt.title("B-splines composed with sine functions")
for j in range(model.exog.shape[1]):
    plt.plot(model.exog[:, j], '-')
pdf.savefig()

# Fit a model with trend and cycling at multiple time scales.  This model has systematic poor fit.
fml = "system_demand_actual ~ bs(q0(date), 5) + bs(f1(date), 5) + bs(f2(date), 5) + (bs(g1(date), 7) + bs(g2(date), 7)) + (bs(h1(date), 12) + bs(h2(date), 12))"
model = sm.OLS.from_formula(fml, data=df) # Don't fit the model

# Basis functions for hourly pattern
for vn in ("h1", "h2"):
    plt.clf()
    ii = [j for j,x in enumerate(model.exog_names) if vn in x]
    for i in ii:
        y = model.exog[0:48, i]
        z = np.linspace(0, 24, len(y))
        plt.plot(z, y, '-', color='grey', alpha=0.8, rasterized=True)
    plt.xlabel("Hour")
    plt.ylabel("Basis function value")
    if vn == "h1":
        plt.title("Cosine functions")
    else:
        plt.title("Sine functions")
    plt.xlim(0, 24)
    pdf.savefig()

# Basis functions for weekly pattern
for vn in ("g1", "g2"):
    plt.clf()
    ii = [j for j,x in enumerate(model.exog_names) if vn in x]
    for i in ii:
        y = model.exog[0:48*7, i]
        z = np.linspace(0, 7, len(y))
        plt.plot(z, y, '-', color='grey', alpha=0.8, rasterized=True)
    plt.xlabel("Day")
    plt.ylabel("Basis function value")
    if vn == "g1":
        plt.title("Cosine functions")
    else:
        plt.title("Sine functions")
    plt.xlim(0, 7)
    pdf.savefig()

# This model doesn't show much systematic misfit.
fml = "system_demand_actual ~ bs(q0(date), 5) + bs(f1(date), 5) + bs(f2(date), 5) + (bs(g1(date), 7) + bs(g2(date), 7)) * (bs(h1(date), 12) + bs(h2(date), 12))"
model = sm.OLS.from_formula(fml, data=df)
result = model.fit_regularized(L1_wt=0, alpha=0.000001)


# Fitted values
dx0 = pd.DataFrame({"date": pd.date_range(start=df.date.min(), end=df.date.max(), freq='H')})
trend = result.predict(exog=dx0)

years = mdates.YearLocator()
halfyears = mdates.YearLocator()
months = mdates.MonthLocator([1, 4, 7, 10])
wdays = mdates.DayLocator()
hours = mdates.HourLocator([0, 3, 6, 9, 12, 15, 18, 21, 24])
yearsFmt = mdates.DateFormatter('%Y')
monthsFmt = mdates.DateFormatter('%b %Y')
wdaysFmt = mdates.DateFormatter('%a')
hoursFmt = mdates.DateFormatter('%H')

plt.clf()
ax = plt.axes()
ax.xaxis.set_major_locator(halfyears)
ax.xaxis.set_major_formatter(monthsFmt)
ax.plot(dx0.date, trend, alpha=0.5)
ax.set_ylim(4000, 7000)
plt.ylabel("Electricity demand")
plt.xlabel("Date")
pdf.savefig()

plt.clf()
ax = plt.axes()
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(monthsFmt)
ax.plot(dx0.date, trend, alpha=0.75)
plt.xlim(pd.to_datetime("2013-01-01"), pd.to_datetime("2014-01-01"))
ax.set_ylim(4000, 7000)
plt.ylabel("Electricity demand")
plt.xlabel("Date")
pdf.savefig()

plt.clf()
ax = plt.axes()
ax.xaxis.set_major_locator(wdays)
ax.xaxis.set_major_formatter(wdaysFmt)
ax.plot(dx0.date, trend)
plt.xlim(pd.to_datetime("2013-03-11"), pd.to_datetime("2013-03-18"))
ax.set_ylim(4000, 7000)
plt.xlabel("Day of week")
plt.ylabel("Electricity demand")
pdf.savefig()

plt.clf()
ax = plt.axes()
ax.xaxis.set_major_locator(hours)
ax.xaxis.set_major_formatter(hoursFmt)
ax.plot(dx0.date, trend)
plt.ylabel("Electricity demand")
plt.xlim(pd.to_datetime("2013-03-11"), pd.to_datetime("2013-03-12"))
ax.set_ylim(4000, 7000)
plt.xlabel("Time of day")
plt.ylabel("Electricity demand")
pdf.savefig()

# Plot of fitted and actual values
for k in range(10):
    year = np.random.choice([2013, 2014, 2015])
    month = np.random.choice(range(1, 13))
    day = np.random.choice(range(1, 20))
    d1 = "%4d-%02d-%02d" % (year, month, day)
    d2 = "%4d-%02d-%02d" % (year, month, day+7)
    plt.clf()
    ax = plt.axes()
    ax.xaxis.set_major_locator(wdays)
    ax.xaxis.set_major_formatter(wdaysFmt)
    ax.plot(df.date, result.fittedvalues)
    ax.plot(df.date, model.endog)
    plt.xlim(pd.to_datetime(d1), pd.to_datetime(d2))
    ax.set_ylim(4000, 7000)
    plt.xlabel("Day of week")
    plt.ylabel("Electricity demand")
    plt.title("%s to %s" % (d1, d2))
    pdf.savefig()

# Residual analyses
res = model.endog - result.fittedvalues

resid = pd.DataFrame({"resid": res, "date": df.date})
resid["dayofyear"] = resid.date.dt.dayofyear
resid["dayofweek"] = resid.date.dt.dayofweek
resid["hourofday"] = resid.date.dt.hour

plt.clf()
x = resid.groupby("dayofweek")["resid"].agg([np.mean, np.std])
plt.plot(x.index, x["mean"], '-', color='blue', lw=4)
plt.plot(x.index, x["mean"] - 2*x["std"], '-', color='orange', lw=2)
plt.plot(x.index, x["mean"] + 2*x["std"], '-', color='orange', lw=2)
pdf.savefig()

plt.clf()
x = resid.groupby("hourofday")["resid"].agg([np.mean, np.std])
plt.plot(x.index, x["mean"], '-', color='blue', lw=4)
plt.plot(x.index, x["mean"] - 2*x["std"], '-', color='orange', lw=2)
plt.plot(x.index, x["mean"] + 2*x["std"], '-', color='orange', lw=2)
pdf.savefig()


df["MeanTemp"] = df['Mean Temperature (°C)']

df["RainTotal"] = df["Daily Rainfall Total (mm)"]

fml = "system_demand_actual ~ bs(q0(date), 5) + bs(f1(date), 5) + bs(f2(date), 5) + (bs(g1(date), 7) + bs(g2(date), 7)) * (bs(h1(date), 12) + bs(h2(date), 12)) + bs(MeanTemp, 5)"
model2 = sm.OLS.from_formula(fml, data=df)
result2 = model2.fit_regularized(L1_wt=0, alpha=0.000001)


pdf.close()
