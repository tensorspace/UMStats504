"""
This script merges weather and electricity usage data for Singapore into a single file.

Before running this script, run compile_weather.py to create the
compiled weather data, and download the electricity data from here:

https://data.gov.sg/dataset/half-hourly-system-demand
"""

import pandas as pd

d1 = pd.read_csv("half-hourly-system-demand-data-from-2-feb-2012-onwards.csv")
d2 = pd.read_csv("weather.csv")

df = pd.merge(d1, d2, left_on="date", right_on="Date")

df = df.drop("Date", axis=1)

df.to_csv("sg_electricity.csv", index=False)
