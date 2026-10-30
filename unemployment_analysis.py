
# CodeAlpha Data Science Internship
# Task 2: Unemployment Analysis with Python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("Unemployment_Rate_upto_11_2020.csv")

# Clean Column Names
df.columns = df.columns.str.strip()

# Display Dataset
print("First 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

# Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Remove Duplicate Records
df = df.drop_duplicates()

# Remove Rows with Missing Values
df = df.dropna()

# Convert Date Column
df["Date"] = pd.to_datetime(
    df["Date"], dayfirst=True
)

# Basic Statistics
print("\nStatistical Summary:")
print(df.describe())

# Average Unemployment Rate by Region
region_avg = df.groupby("Region")[
    "Estimated Unemployment Rate (%)"
].mean().sort_values(ascending=False)

print("\nAverage Unemployment by Region:")
print(region_avg)

# Graph 1: Regional Unemployment
plt.figure(figsize=(12, 6))

sns.barplot(
    x=region_avg.values,
    y=region_avg.index
)

plt.title("Average Unemployment Rate by Region")
plt.xlabel("Unemployment Rate (%)")
plt.ylabel("Region")
plt.tight_layout()
plt.savefig("regional_unemployment.png")
plt.show()

# Graph 2: Unemployment Over Time
monthly = df.groupby("Date")[
    "Estimated Unemployment Rate (%)"
].mean()

plt.figure(figsize=(12, 6))
plt.plot(
    monthly.index,
    monthly.values,
    marker="o"
)

plt.title("Unemployment Rate Over Time")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig("unemployment_trend.png")
plt.show()

# Graph 3: Unemployment Distribution
plt.figure(figsize=(10, 6))

sns.histplot(
    df["Estimated Unemployment Rate (%)"],
    bins=20,
    kde=True
)

plt.title("Distribution of Unemployment Rates")
plt.xlabel("Unemployment Rate (%)")
plt.tight_layout()
plt.savefig("unemployment_distribution.png")
plt.show()

# COVID-19 Period Analysis
pre_covid = df[
    df["Date"] < "2020-03-01"
]

covid_period = df[
    df["Date"] >= "2020-03-01"
]

print("\nCOVID-19 Impact Analysis")

if not pre_covid.empty:
    print(
        "Before March 2020:",
        pre_covid["Estimated Unemployment Rate (%)"].mean()
    )
else:
    print("No pre-March 2020 data available.")

print(
    "March 2020 Onward:",
    covid_period["Estimated Unemployment Rate (%)"].mean()
)

print("\nAnalysis Completed Successfully!")
