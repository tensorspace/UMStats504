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
    return np.cos(2 * np.pi * x.dt.dayofyear / 7)
def g2(x):
    return np.sin(2 * np.pi * x.dt.dayofyear / 7)

# Periodic cycle by day
def h1(x):
    return np.cos(2 * np.pi * x.dt.hour / 24)
def h2(x):
    return np.sin(2 * np.pi * x.dt.hour / 24)

# Fit a model with trend and cycling at multiple time scales
fml = "system_demand_actual ~ bs(q0(date), 3) + bs(f1(date), 3) + bs(f2(date), 3) + bs(g1(date), 3) + bs(g2(date), 3) + bs(h1(date), 3) + bs(h2(date), 3)"
model = sm.OLS.from_formula(fml, data=df)
result = model.fit_regularized(L1_wt=0, alpha=0.1)

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
ax.plot(dx0.date, trend)
ax.set_ylim(4000, 6500)
plt.ylabel("Electricity demand")
pdf.savefig()

plt.clf()
ax = plt.axes()
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(monthsFmt)
ax.plot(dx0.date, trend)
plt.xlim(pd.to_datetime("2013-01-01"), pd.to_datetime("2014-01-01"))
ax.set_ylim(4000, 6500)
plt.ylabel("Electricity demand")
pdf.savefig()

plt.clf()
ax = plt.axes()
ax.xaxis.set_major_locator(wdays)
ax.xaxis.set_major_formatter(wdaysFmt)
ax.plot(dx0.date, trend)
plt.xlim(pd.to_datetime("2013-03-11"), pd.to_datetime("2013-03-18"))
ax.set_ylim(4000, 6500)
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
ax.set_ylim(4000, 6500)
plt.xlabel("Time of day")
plt.ylabel("Electricity demand")
pdf.savefig()

pdf.close()
