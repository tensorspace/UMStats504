"""
This script downloads daily weather data for Singapore.  There is a
separate data file for every month.
"""

import os
import os.path

if not os.path.exists("weather_raw"):
    os.makedirs("weather_raw")

for year in range(2012, 2018):
    for month in range(1, 13):
        cmd = ("wget http://www.weather.gov.sg/files/dailydata/DAILYDATA_S24_%d%02d.csv -P weather_raw" %
               (year, month))
        os.system(cmd)
