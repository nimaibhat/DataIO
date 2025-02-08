import pandas as pd
import matplotlib as plt
import seaborn as sns

df = pd.read_csv('globalAirPollution.csv')
aqi_sort= df.sort_values(by="AQI Value", ascending=True)
filtered_nan = aqi_sort[pd.isna(aqi_sort["Country"])]
aqi_sort = aqi_sort.dropna(subset=["Country"])
print(aqi_sort)
