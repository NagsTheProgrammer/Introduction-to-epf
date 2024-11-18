import pandas as pd  # requires: pip install pandas
import torch
import matplotlib.pyplot as plt  # requires: pip install matplotlib
import numpy as np
import requests

from chronos import ChronosPipeline

num_days_predict = 2
num_days_back = 30

# Load the data into a pandas DataFrame
df = pd.read_csv('datasets/aeso_dataset_2022_2023.csv')
df.set_index('Date/Time', inplace=True)

df = df[24 * (-30):]

# Print the first few rows of the dataframe
print("First 5 rows of the dataframe:")
print(df.head())

# Print basic statistical summary
print("\nStatistical summary of the dataframe:")
print(df.describe())

# Print data types and check for missing values
print("\nData types and missing values information:")
print(df.info())

pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cpu",  # use "cpu" for CPU inference and "mps" for Apple Silicon
    torch_dtype=torch.bfloat16,
)

forecast = pipeline.predict(
    context=torch.tensor(df["Price"]),
    prediction_length=24 * num_days_predict,
    num_samples=24 * num_days_back,

)

forecast_index = range(len(df), len(df) + 24 * num_days_predict)
low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)

plt.figure(figsize=(8, 4))
plt.plot(df["Price"], color="royalblue", label="historical data")
plt.plot(forecast_index, median, color="tomato", label="median forecast")
plt.fill_between(forecast_index, low, high, color="tomato", alpha=0.3, label="80% prediction interval")
plt.legend()
plt.grid()
plt.show()