import pandas as pd
import numpy as np

datasets = ['price', 'ail', 'wind', 'solar']
merged_df = pd.DataFrame()

for dataset in datasets:
    filename = f'datasets/aeso_{dataset}_2022_2023.csv'

    df = pd.read_csv(filename)

    df.set_index('Date/Time', inplace=True)
    df.index = pd.to_datetime(df.index, format='%m/%d/%Y %H:%M')

    merged_df = pd.concat([merged_df, df], axis=1)

print("Merged DataFrame:")
print(merged_df.head())

# Check for missing data in the merged DataFrame
missing_data = merged_df.isna().sum()

# Print columns with missing data and the number of missing entries
print("Missing data in each column:")
print(missing_data[missing_data > 0])

# Check the total number of missing values in the entire DataFrame
total_missing = merged_df.isna().sum().sum()
print(f"\nTotal number of missing values: {total_missing}")

merged_df.interpolate(method='linear', inplace=True)  # Linear interpolation

# Check the total number of missing values in the entire DataFrame
total_missing = merged_df.isna().sum().sum()
print(f"\nTotal number of missing values: {total_missing}")

merged_df.to_csv('datasets/aeso_dataset_2022_2023.csv')
