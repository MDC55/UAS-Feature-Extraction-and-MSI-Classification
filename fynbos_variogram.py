#!/usr/bin/env python
# coding: utf-8

# In[6]:


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


# In[12]:


#stacking Green, Red, Red Edge, and Near Infrared band individual rasters
#into one multi-band raster
#top_path='/gdrive/My Drive/Fynbos/October_2023/Grootbos_Drone_fil/s/mavic3m/'
top_path='S:/mc5545/SA_Drone_data/'
raster_path = top_path+'burnplot18_lr/reflectance'
raster_filename = 'burnplot18_lr_band_stack.tif'
#load shapefile
shp = gpd.read_file(Path(top_path,'burnplot18_lr/burnplot18_lr_shape_all/burnplot18_lr_shp.shp'))


# In[13]:

#load raster
raster = rxr.open_rasterio(Path(raster_path, raster_filename),masked=True).squeeze()

geometries = shp.geometry.apply(mapping)


#%%
#clipped plot for first 10
# Create an empty list to store the clipped rasters
clipped_rasters = []

for i in range(len(shp)):
#for i in range(1):    
  # Clip the raster with the current geometry
  p1 = raster.rio.clip([geometries[i]],shp.crs)

  # Convert the clipped raster to a NumPy array
  p1_np = np.asarray(p1)

  # Append the clipped raster array to the list
  clipped_rasters.append(p1_np)
  


#%%

import numpy as np

# Assuming clipped_rasters is your list of numpy arrays with shape (4, 553, 552)

# Initialize lists to store mean values for each band
mean_values_band1 = []
mean_values_band2 = []
mean_values_band3 = []
mean_values_band4 = []

# Loop through each raster
for raster in clipped_rasters:
    # Calculate mean for each band and append to the respective list
    mean_values_band1.append(np.nanmean(raster[0]))
    mean_values_band2.append(np.nanmean(raster[1]))
    mean_values_band3.append(np.nanmean(raster[2]))
    mean_values_band4.append(np.nanmean(raster[3]))

# Now, mean_values_band1, mean_values_band2, mean_values_band3, mean_values_band4
# contain the mean values for each band across all clipped rasters



# Combine mean values into a single NumPy array
mean_values = np.vstack((mean_values_band1, mean_values_band2, 
                                     mean_values_band3, mean_values_band4))



# Initialize lists to store standard deviation values for each band
std_band1 = []
std_band2 = []
std_band3 = []
std_band4 = []
# Loop through each raster
for raster in clipped_rasters:
    # Calculate standard deviation for each band and append to the respective list
    std_band1.append(np.nanstd(raster[0]))
    std_band2.append(np.nanstd(raster[1]))
    std_band3.append(np.nanstd(raster[2]))
    std_band4.append(np.nanstd(raster[3]))     
    # Combine standard deviation values into a single NumPy array
std_values = np.vstack((std_band1, std_band2,std_band3, std_band4))


from scipy.stats import skew, kurtosis

# Initialize lists to store skewness values for each band
skewness_band1 = []
skewness_band2 = []
skewness_band3 = []
skewness_band4 = []

# Initialize lists to store kurtosis values for each band
kurtosis_band1 = []
kurtosis_band2 = []
kurtosis_band3 = []
kurtosis_band4 = []

# Loop through each raster
for raster in clipped_rasters:

        
    # Calculate various statistics for each channel, ignoring nan values
    mean = np.nanmean(raster[0], axis=(0, 1))
    std = np.nanstd(raster[0], axis=(0, 1))
    q1 = np.nanpercentile(raster[0], 25, axis=(0, 1))  # First quartile
    q3 = np.nanpercentile(raster[0], 75, axis=(0, 1))  # Third quartile
    skewness_ = np.nanmean((raster[0] - mean)**3, axis=(0, 1)) / std**3
    kurtosis_ = np.nanmean((raster[0] - mean)**4, axis=(0, 1)) / std**4 - 3 
    
    #Or
    
    # Calculate skewness for each band and append to the respective list
    skewness_band1.append(skew(raster[0],axis=None, nan_policy='omit'))
    skewness_band2.append(skew(raster[1],axis=None, nan_policy='omit'))
    skewness_band3.append(skew(raster[2],axis=None, nan_policy='omit'))
    skewness_band4.append(skew(raster[3],axis=None, nan_policy='omit'))

    # Calculate kurtosis for each band and append to the respective list
    kurtosis_band1.append(kurtosis(raster[0],axis=None, nan_policy='omit'))
    kurtosis_band2.append(kurtosis(raster[1], axis=None,nan_policy='omit'))
    kurtosis_band3.append(kurtosis(raster[2], axis=None,nan_policy='omit'))
    kurtosis_band4.append(kurtosis(raster[3],axis=None, nan_policy='omit'))
 

# Combine skewness values into a single NumPy array
skewness_values = np.vstack((skewness_band1, skewness_band2, skewness_band3, skewness_band4))

# Combine kurtosis values into a single NumPy array
kurtosis_values = np.vstack((kurtosis_band1, kurtosis_band2, kurtosis_band3, kurtosis_band4))

    

# Now, mean_values_array is a NumPy array where each column represents the mean values for a specific band
#l=mean_values[0]
    # Iterate through the clipped rasters
#%%

from scipy.stats import skew, kurtosis

# Create an empty list to store the indices values for each clipped raster (38 total)
NDVI_values = []

# Iterate through the clipped rasters
for p1_np in clipped_rasters:
    # Calculate NDVI
    NDVI = (p1_np[3] - p1_np[1]) / (p1_np[3] + p1_np[1])
    
    # Append the indices value to the list
    NDVI_values.append(NDVI)

# Calculate mean, standard deviation, skewness, and kurtosis for NDVI values
mean_NDVI = [np.nanmean(v) for v in NDVI_values]
std_NDVI = [np.nanstd(v) for v in NDVI_values]
skewness_NDVI = [skew(v, axis=None, nan_policy='omit') for v in NDVI_values]
kurtosis_NDVI = [kurtosis(v, axis=None, nan_policy='omit') for v in NDVI_values]
CV_NDVI=std_NDVI/mean_NDVI
# Calculate Coefficient of Variation (CV) for NDVI
CV_NDVI = [std / mean if mean != 0 else np.nan for std, mean in zip(std_NDVI, mean_NDVI)]


#%%

shp.crs
shp.plot()

# Assuming 'gdf' is your GeoDataFrame
# Replace 'geometry' with the actual name of your geometry column
centroid =shp.geometry.centroid
centroid.dtypes
# Assuming 'shp_reprojected' is your GeoDataFrame

# Extract X and Y coordinates from the centroids
# x=centroid.x.astype(int)
# y=centroid.y.astype(int)
# centroid.x.round(2)

df = pd.DataFrame({"x": centroid.x, 
                   "y": centroid.y,
                   "mean":mean_values[3]})
# Now 'centroid_x' and 'centroid_y' columns contain integer coordinates

#%%
# Calculation variogram
V = skg.Variogram(coordinates=df[['x', 'y']].values, values=df['mean'].values)
print(V)

# Variogram visualization
V.plot()
plt.close

# SILL: The value at which the model first flattens out. 
# RANGE: The distance at which the model first flattens out.
# NUGGET: The value at which the semi-variogram (almost) intercepts the y-value

#%% converting shape files crs


# import geopandas as gpd
# # Assuming 'shp' is your GeoDataFrame
# # Replace 'current_crs' with the current CRS of your GeoDataFrame
# # Replace 'target_crs' with the target CRS you want to reproject to

# # Check the current CRS
# print("Current CRS:", shp.crs)

# # Replace 'target_crs' with the CRS you want to reproject to
# target_crs = 'EPSG:4326'  # Example: WGS84
# # Reproject to the target CRS
# shp_reprojected = shp.geometry.to_crs(target_crs)

# # Check the new CRS
# print("Reprojected CRS:", shp_reprojected.crs)

# d=shp_reprojected.centroid

# # Calculate centroids
# shp_reprojected['centroid'] = shp_reprojected.centroid

# # Now 'shp_reprojected' contains the GeoDataFrame in the new CRS with centroid information

# #%%
# #gpd.show_versions()


# #%%




# #%% Extracting Lat Long from the whole image/raster and then calculating the variogram
# #%% test clip of first plot "p1"
# i = 0
# p1 = raster.rio.clip([geometries[i]],shp.crs)
# p1_np = np.asarray(p1)



# from rasterio.warp import transform

# print('source raster crs',raster.rio.crs)
# raster_crs = raster.rio.reproject('EPSG:4326')

# da = p1
# ny, nx = len(da['y']), len(da['x'])
# x, y = np.meshgrid(da['x'], da['y'])
# # Rasterio works with 1D arrays
# lon, lat = transform(da.rio.crs, {'init': 'EPSG:4326'},
#                               x.flatten(), y.flatten())
# lon = np.asarray(lon).reshape((ny, nx))
# lat = np.asarray(lat).reshape((ny, nx))
# da.coords['lon'] = (('y', 'x'), lon)
# da.coords['lat'] = (('y', 'x'), lat)
# l1 = lat.flatten()
# l2 = lon.flatten()

# pixel_values=np.asarray(p1)
# c=pixel_values[0].flatten()
# df = pd.DataFrame({"Lat": l1, "Long": l2, "values": c})

# #%%
# # Calculation variogram
# V = skg.Variogram(coordinates=df[['Lat', 'Long']].values, values=df['values'].values)
# print(V)

# #%%
# # Variogram visualization
# V.plot()
# plt.close
