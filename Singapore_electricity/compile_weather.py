"""
This script concatenates monthly weather data for Singapore into a single file.

Before running this script, run download_weather.py to obtain the raw data files.
"""

import pandas as pd
import os
import os.path

files = os.listdir("weather_raw")
files = [x for x in files if x.endswith(".csv")]

for i,f in enumerate(files):
    fi = os.path.join("weather_raw", f)
    if i == 0:
        df = pd.read_csv(fi, encoding='latin1')
    else:
        dx = pd.read_csv(fi, encoding='latin1')
        df = pd.concat((df, dx), axis=0)

cnv = ["Highest 30 Min Rainfall (mm)", "Highest 60 Min Rainfall (mm)",
       "Highest 120 Min Rainfall (mm)", "Mean Wind Speed (km/h)",
       "Max Wind Speed (km/h)"]

for c in cnv:
    df[c] = pd.to_numeric(df[c], errors='coerce')

df["Date"] = df.apply(lambda x: "%4d-%02d-%02d" % (x.Year, x.Month, x.Day), axis=1)
df["Date"] = pd.to_datetime(df.Date)

df = df.drop(["Year", "Month", "Day"], axis=1)

df.to_csv("weather.csv", index=False)
