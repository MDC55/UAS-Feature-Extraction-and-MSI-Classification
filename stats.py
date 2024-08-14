# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 13:53:28 2023

@author: mc5545
"""
#importing packages
import os
os.environ['USE_PYGEOS'] = '0'
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import fiona
from fiona import Feature, Geometry
from shapely.geometry import mapping
import rioxarray as rxr
import xarray as xr
import geopandas as gpd
import rasterio as rio
from rasterio.enums import Resampling

import skgstat as skg

# Load DataFrame from the CSV file
directory_path='S:/mc5545/SA_Drone_data/Indices_csv/'
df1 = pd.read_csv(f'{directory_path}indices_2006_burnplot18.csv', index_col=0)
df2 = pd.read_csv(f'{directory_path}indices_2016_burn2016.csv', index_col=0)
df3 = pd.read_csv(f'{directory_path}indices_2017_burn2017.csv', index_col=0)
df4 = pd.read_csv(f'{directory_path}indices_2019_burn2019.csv', index_col=0)
df5 = pd.read_csv(f'{directory_path}indices_2020_burnplot17.csv', index_col=0)
df6 = pd.read_csv(f'{directory_path}indices_2022_burn2022.csv', index_col=0)


dataframes = [df1, df2, df3, df4, df5, df6]
#%%
# Assuming dataframes is your list of dataframes
ndvi_values = [df['NDVI'].dropna().values for df in dataframes]

ndvi_std_values = [df['Std_NDVI'].dropna().values for df in dataframes]
# Calculate the mean of 'NDVI' for each DataFrame
mean_ndvi_values = [np.mean(ndvi) for ndvi in ndvi_values]
mean_std_values1 = [np.std(ndvi) for ndvi in ndvi_values]

mean_std_values2 = [np.mean(ndvi) for ndvi in ndvi_std_values]
years = [2006, 2016, 2017, 2019, 2020, 2022]




#%%ReCI
ndvi_values = [df['ReCI'].dropna().values for df in dataframes]
mean_std_values1 = [np.std(ndvi) for ndvi in ndvi_values]

#%%DVI
ndvi_values = [df['DVI'].dropna().values for df in dataframes]
mean_std_values1 = [np.std(ndvi) for ndvi in ndvi_values]
#%% GSAVI
ndvi_values = [df['GSAVI'].dropna().values for df in dataframes]
mean_std_values1 = [np.std(ndvi) for ndvi in ndvi_values]
#%% MSAVI2
ndvi_values = [df['MSAVI2'].dropna().values for df in dataframes]
mean_std_values1 = [np.std(ndvi) for ndvi in ndvi_values]
#%% CVI
ndvi_values = [df['CVI'].dropna().values for df in dataframes]
mean_std_values1 = [np.std(ndvi) for ndvi in ndvi_values]

#%%
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import seaborn as sns

# Assuming you have the following variables defined:
# mean_ndvi_values, mean_std_values2, and years

# Reshape the data to a 2D array as required by scikit-learn
X = np.array(mean_ndvi_values).reshape(-1, 1)
y = np.array(years)

# Create a linear regression model with sample weights
regressor = LinearRegression()
regressor.fit(X, y, sample_weight=mean_std_values1)
#regressor.fit(X, y)
# Predict the years for new data (using the mean_ndvi_values as features)
predicted_years = regressor.predict(X)

# Display the coefficients and intercept
print("Coefficients:", regressor.coef_)
print("Intercept:", regressor.intercept_)

# Display the predicted years
print("Predicted Years:", predicted_years)

# Calculate R-squared (R²)
r_squared = r2_score(y, predicted_years)

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(y, predicted_years))

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y, predicted_years)

# Plot scatter plot with error bars
plt.figure(figsize=(10, 6))
plt.errorbar(years, mean_ndvi_values, yerr=mean_std_values1, capsize=4,fmt='o', color='blue', label='Actual with Error Bars')
plt.plot(predicted_years, mean_ndvi_values, color='red', label='Weighted Regression Line')
plt.xlabel('Year')
plt.ylabel('NDVI Values')
plt.legend()
# Display metrics on the graph
metrics_text = f'R-squared: {r_squared:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}'
plt.text(0.05, 0.85, metrics_text, transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))

plt.grid()
plt.show()

#%%

#%%

import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Your data
years = [2006, 2016, 2017, 2019, 2020, 2022]
mean_ndvi_values = [np.mean(ndvi) for ndvi in ndvi_values]
weight = mean_std_values1

x = np.array(mean_ndvi_values).reshape(-1, 1)
y = np.array(years)

# Calculate weights based on mean and standard deviation
weights = 1 / np.sqrt(np.std(y))  # You can adjust this based on your specific criteria

# Add a constant term to the independent variable matrix
X = sm.add_constant(x)

# Perform weighted linear regression
model = sm.WLS(y, X, weights=weights)
results = model.fit()

# Extract R-squared value from results
rsq = results.rsquared

# Predict values using the model
y_pred = results.predict(X)

# Calculate other statistics
rmse = np.sqrt(mean_squared_error(y, y_pred))
mae = mean_absolute_error(y, y_pred)

# Display regression results
print(results.summary())
print(f"R-squared: {rsq:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

# Plot the results
import matplotlib.pyplot as plt

plt.errorbar(years, mean_ndvi_values, yerr=mean_std_values1, capsize=4, fmt='o', color='blue', label='Actual with Error Bars')
plt.plot(y_pred,x, color='red', label='Weighted Regression Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid()

# Annotate the plot with statistics
plt.annotate(f'R-squared: {rsq:.4f}', xy=(0.06, 0.85), xycoords='axes fraction', fontsize=10)
plt.annotate(f'RMSE: {rmse:.4f}', xy=(0.06, 0.80), xycoords='axes fraction', fontsize=10)
plt.annotate(f'MAE: {mae:.4f}', xy=(0.06, 0.75), xycoords='axes fraction', fontsize=10)

plt.show()
#%% Confidence Interval not working

import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Your data
years = [2006, 2016, 2017, 2019, 2020, 2022]
mean_ndvi_values = [np.mean(ndvi) for ndvi in ndvi_values]
mean_std_values1 = [np.std(ndvi) for ndvi in ndvi_values]
weight = [np.mean(ndvi) for ndvi in ndvi_std_values]
x = np.array(mean_ndvi_values).reshape(-1, 1)
y = np.array(years)

# Calculate weights based on mean and standard deviation
weights = 1 / np.sqrt(np.std(y))  # You can adjust this based on your specific criteria

# Add a constant term to the independent variable matrix
X = sm.add_constant(x)

# Perform weighted linear regression
model = sm.WLS(y, X, weights=weights)
results = model.fit()

# Extract R-squared value and confidence intervals from results
rsq = results.rsquared
conf_int = results.conf_int()

# Predict values using the model
y_pred = results.predict(X)

# Calculate other statistics
rmse = np.sqrt(mean_squared_error(y, y_pred))
mae = mean_absolute_error(y, y_pred)

# Display regression results
print(results.summary())
print(f"R-squared: {rsq:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

# Plot the results with confidence interval
plt.errorbar(years, mean_ndvi_values, yerr=mean_std_values1, capsize=4, fmt='o', color='blue', label='Actual with Error Bars')
plt.plot(y_pred,x, color='red', label='Weighted Regression Line')

# Plot confidence interval
plt.fill_between(y, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='gray', alpha=0.3, label='95% Confidence Interval')

plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid()

# Annotate the plot with statistics
plt.annotate(f'R-squared: {rsq:.4f}', xy=(0.05, 0.85), xycoords='axes fraction', fontsize=10)
plt.annotate(f'RMSE: {rmse:.4f}', xy=(0.05, 0.80), xycoords='axes fraction', fontsize=10)
plt.annotate(f'MAE: {mae:.4f}', xy=(0.05, 0.75), xycoords='axes fraction', fontsize=10)

plt.show()
#%%
dataframes = [df1, df2, df3, df4, df5, df6]
# Assuming dataframes is your list of dataframes
ndvi_values = [df['NDVI'].dropna().values for df in dataframes]
ndvi_std_values = [df['Std_NDVI'].dropna().values for df in dataframes]

# Calculate the mean and std of vegetation index for each DataFrame
mean_VI = [np.mean(ndvi) for ndvi in ndvi_values]
std_VI = [np.std(ndvi) for ndvi in ndvi_values]


mean_std_values2 = [np.mean(ndvi) for ndvi in ndvi_std_values]

years = [2006, 2016, 2017, 2019, 2020, 2022]

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# Reshape the data to a 2D array as required by scikit-learn
X = np.array(mean_VI).reshape(-1, 1)
y = np.array(years)

# Create a linear regression model with sample weights
regressor = LinearRegression()
regressor.fit(X, y, sample_weight=std_VI)
#regressor.fit(X, y)
# Predict the years for new data (using the mean_VI as features)
predicted_years = regressor.predict(X)

# Display the coefficients and intercept
print("Coefficients:", regressor.coef_)
print("Intercept:", regressor.intercept_)

# Display the predicted years
print("Predicted Years:", predicted_years)

# Calculate R-squared (R²)
r_squared = r2_score(y, predicted_years)

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(y, predicted_years))

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y, predicted_years)

# Plot scatter plot with error bars
plt.figure(figsize=(10, 6))
plt.errorbar(years, mean_VI, yerr=std_VI, capsize=4,fmt='o', color='blue', label='Actual with Error Bars')
plt.plot(predicted_years, mean_VI, color='red', label='Weighted Regression Line')
plt.xlabel('Year')
plt.ylabel('NDVI Values')
plt.legend()
# Display metrics on the graph
metrics_text = f'R-squared: {r_squared:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}'
plt.text(0.05, 0.85, metrics_text, transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))

plt.grid()
plt.show()


#%%  final
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

def perform_regression(dataframes, column_name):
    # Extract NDVI values and standard deviations for the specified column from each dataframe
    VI_values = [df[column_name].dropna().values for df in dataframes]
    #VI_std_values = [df['Std_' + column_name].dropna().values for df in dataframes]
    #std_VI1 = [np.mean(ndvi) for ndvi in VI_std_values]

    # Calculate the mean and std of vegetation index for each DataFrame
    mean_VI = [np.mean(ndvi) for ndvi in VI_values]
    std_VI = [np.std(ndvi) for ndvi in VI_values]

    years = [2006, 2016, 2017, 2019, 2020, 2022]
    
    # Reshape the data to a 2D array as required by scikit-learn
    X = np.array(mean_VI).reshape(-1, 1)
    y = np.array(years)

    # Create a linear regression model with sample weights
    regressor = LinearRegression()
    #regressor.fit(X, y, sample_weight=std_VI)
    regressor.fit(X, y)

    # Predict the years for new data (using the mean_VI as features)
    predicted_years = regressor.predict(X)

    # Display the coefficients and intercept
    print("Coefficients:", regressor.coef_)
    print("Intercept:", regressor.intercept_)

    # Display the predicted years
    print("Predicted Years:", predicted_years)

    # Calculate R-squared (R²)
    r_squared = (r2_score(y, predicted_years))

    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mean_squared_error(y, predicted_years))

    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(y, predicted_years)

    # Plot scatter plot with error bars
    plt.figure(figsize=(10, 6))
    plt.errorbar(years, mean_VI, yerr=std_VI, capsize=4, fmt='o', color='blue', label=f'Actual {column_name} with Error Bars')
    plt.plot(predicted_years, mean_VI, color='red', label='Weighted Regression Line')
    plt.xlabel('Year')
    plt.ylabel(f'{column_name} Values')
    plt.legend(loc='upper right')
    

    # Display metrics on the graph
    metrics_text = f'R-squared: {r_squared:.2f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}'
    plt.text(0.05, 0.85, metrics_text, transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))

    plt.grid()
    plt.show()

# Example usage
dataframes = [df1, df2, df3, df4, df5, df6]
years = [2006, 2016, 2017, 2019, 2020, 2022]
column_name = 'std_NIR'  # Specify the column name you want to analyze

perform_regression(dataframes, column_name)

#%%
# Assuming dataframes is your list of dataframes
ndvi_values = [df['NDVI'].dropna().values for df in dataframes]

# Calculate the mean of 'NDVI' for each DataFrame
mean_ndvi_values = [np.mean(ndvi) for ndvi in ndvi_values]
mean_std_values1 = [np.std(ndvi) for ndvi in ndvi_values]

years = [2006, 2016, 2017, 2019, 2020, 2022]

