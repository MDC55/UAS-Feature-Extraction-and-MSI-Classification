#!/usr/bin/env python
# coding: utf-8

# In[1]:


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




#stacked Green, Red, Red Edge, and Near Infrared band individual rasters into one multi-band raster
top_path='S:/mc5545/SA_Drone_data/'

raster_path1 = top_path+'burnplot18_lr/reflectance'
raster_filename1 = 'burnplot18_lr_band_stack.tif'

raster_path = top_path+'burnplot18_lr/reflectance'
raster_filename = 'burnplot18_lr_band_stack.tif'
    
# # Small ROI



shapefiles = gpd.read_file(Path(top_path,'burnplot18_lr/burnplot18_lr_shape/burnplot18_lr_shp.shp'))
shapefiles1 = gpd.read_file(Path(top_path,'burnplot18_lr/burnplot18_lr_shape/burnplot18_lr_shp.shp'))





#%%  now
# load raster1
def fynbos_texture(raster_path, raster_filename, shapefiles):
    # Open the raster using rasterio
    raster = rxr.open_rasterio(Path(raster_path, raster_filename), masked=True).squeeze()
    geometries = shapefiles.geometry.apply(mapping)
    
    # Create an empty list to store the clipped rasters
    clipped_rasters = []
    for i in range(4):
        # Clip the raster with the current geometry
        p1 = raster.rio.clip([geometries[i]], shapefiles.crs)
        # Convert the clipped raster to a NumPy array
        p1_np = np.asarray(p1)
        # Append the clipped raster array to the list
        clipped_rasters.append(p1_np)

    # Import necessary libraries and modules
    import sys
    from texture import fastglcm_wrapper
    sys.path.append('/S:/mc5545/SA_Drone_data')
    from scipy.stats import skew, kurtosis
    
    # Initialize an empty list to store results for all rasters
    data_list  = []
    data_list1 = []
    data_list2 = [] 
    data_list3 = []
    data_list4 = [] 
    data_list5 = []
    data_list6 = []
    data_list7 = []
    data_list8 = []
    # Loop through each raster
    for idx, raster in enumerate(clipped_rasters):
        # Transpose the raster to have the band order (G, R, RE, NIR)
        raster = np.transpose(raster, (1, 2, 0))
        
        # Create an instance of the fastglcm_wrapper class
        # Specify the parameters: levels, kernel_size, distance_offset, and angles
        tex2 = fastglcm_wrapper(raster, levels=8, kernel_size=5, distance_offset=5, angles=[0, 45, 90, 135])
       
        # Calculate various GLCM matrices
        mean = tex2.calculate_glcm_mean()
        variance = tex2.calculate_glcm_var()
        contrast = tex2.calculate_glcm_contrast()
        dissimilarity = tex2.calculate_glcm_dissimilarity()
        homogeneity = tex2.calculate_glcm_homogenity()
        asm = tex2.calculate_glcm_asm()
        entropy = tex2.calculate_glcm_entropy()
        maximum = tex2.calculate_glcm_max()
        correlation = tex2.calculate_glcm_correlation()

        # Function to calculate statistics for each GLCM matrix
        def calculate_stats(tex_features, prefix):
            mean_tex_features = np.nanmean(tex_features, axis=(0, 1))
            std_tex_features = np.nanstd(tex_features, axis=(0, 1))
            skewness_tex_features = skew(tex_features, axis=(0, 1), nan_policy='omit')
            kurtosis_tex_features = kurtosis(tex_features, axis=(0, 1), nan_policy='omit')
            CV_tex_features = (std_tex_features / mean_tex_features)
            
            # Create dictionary keys with the provided prefix
            keys = [f'{prefix}_{param}' for param in ['mean', 'std', 'skewness', 'kurtosis', 'CV']]
            # Inside the calculate_stats function, before returning the dictionary
            print(f'Keys: {keys}')
            print(f'Dictionary: {dict(zip(keys, [mean_tex_features, std_tex_features, skewness_tex_features, kurtosis_tex_features, CV_tex_features]))}')

            # Return results in a dictionary
            return dict(zip(keys, [mean_tex_features, std_tex_features, skewness_tex_features, 
                                   kurtosis_tex_features, CV_tex_features]))

        # Calculate statistics for each GLCM matrix and append the results to the list
        data_list.append(calculate_stats(mean, 'mean'))
        data_list1.append(calculate_stats(variance, 'variance'))
        data_list2.append(calculate_stats(contrast, 'contrast'))
        data_list3.append(calculate_stats(dissimilarity, 'dissimilarity'))
        data_list4.append(calculate_stats(homogeneity, 'homogeneity'))
        data_list5.append(calculate_stats(asm, 'asm'))
        data_list6.append(calculate_stats(entropy, 'entropy'))
        data_list7.append(calculate_stats(maximum, 'maximum'))
        data_list8.append(calculate_stats(correlation, 'correlation'))

    # Extract and organize values into arrays
    parameters = ['mean_CV', 'mean_kurtosis', 'mean_mean', 'mean_skewness', 'mean_std']
    parameters1 = ['variance_CV', 'variance_kurtosis', 'variance_mean', 'variance_skewness', 'variance_std']
    parameters2 = ['contrast_CV', 'contrast_kurtosis', 'contrast_mean', 'contrast_skewness', 'contrast_std']
    parameters3 = ['dissimilarity_CV', 'dissimilarity_kurtosis', 'dissimilarity_mean', 'dissimilarity_skewness', 'dissimilarity_std']
    parameters4 = ['homogeneity_CV', 'homogeneity_kurtosis', 'homogeneity_mean', 'homogeneity_skewness', 'homogeneity_std']
    parameters5 = ['asm_CV', 'asm_kurtosis', 'asm_mean', 'asm_skewness', 'asm_std']
    parameters6 = ['entropy_CV', 'entropy_kurtosis', 'entropy_mean', 'entropy_skewness', 'entropy_std']
    parameters7 = ['maximum_CV', 'maximum_kurtosis', 'maximum_mean', 'maximum_skewness', 'maximum_std']
    parameters8 = ['correlation_CV', 'correlation_kurtosis', 'correlation_mean', 'correlation_skewness', 'correlation_std']
    bands = ['band1', 'band2', 'band3', 'band4']

    data = {}
    for parameter in parameters:
         for band_index, band in enumerate(bands, start=1):
             key = f"{parameter}_{band}"
             data[key] = np.array([d[parameter][band_index - 1] for d in data_list])
             
    data1 = {}
    for parameter in parameters1:
         for band_index, band in enumerate(bands, start=1):
             key = f"{parameter}_{band}"
             data1[key] = np.array([d[parameter][band_index - 1] for d in data_list1])

    data2 = {}
    for parameter in parameters2:
         for band_index, band in enumerate(bands, start=1):
             key = f"{parameter}_{band}"
             data2[key] = np.array([d[parameter][band_index - 1] for d in data_list2])

    data3 = {}
    for parameter in parameters3:
         for band_index, band in enumerate(bands, start=1):
             key = f"{parameter}_{band}"
             data3[key] = np.array([d[parameter][band_index - 1] for d in data_list3])

    data4 = {}
    for parameter in parameters4:
         for band_index, band in enumerate(bands, start=1):
             key = f"{parameter}_{band}"
             data4[key] = np.array([d[parameter][band_index - 1] for d in data_list4])

    data5 = {}
    for parameter in parameters5:
         for band_index, band in enumerate(bands, start=1):
             key = f"{parameter}_{band}"
             data5[key] = np.array([d[parameter][band_index - 1] for d in data_list5])


    data6 = {}
    for parameter in parameters6:
         for band_index, band in enumerate(bands, start=1):
             key = f"{parameter}_{band}"
             data6[key] = np.array([d[parameter][band_index - 1] for d in data_list6])


    data7 = {}
    for parameter in parameters7:
         for band_index, band in enumerate(bands, start=1):
             key = f"{parameter}_{band}"
             data7[key] = np.array([d[parameter][band_index - 1] for d in data_list7])

    data8 = {}
    for parameter in parameters8:
         for band_index, band in enumerate(bands, start=1):
             key = f"{parameter}_{band}"
             data8[key] = np.array([d[parameter][band_index - 1] for d in data_list8])
             

    # Create a DataFrame
    df = pd.DataFrame(data)
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    df3 = pd.DataFrame(data3)
    df4 = pd.DataFrame(data4)
    df5 = pd.DataFrame(data5)
    df6 = pd.DataFrame(data6)
    df7 = pd.DataFrame(data7)
    df8 = pd.DataFrame(data8)
    
    # Concatenate them along the columns (axis=1)
    result_df = pd.concat([df, df1, df2, df3, df4, df5, df6, df7, df8], axis=1)
    # Print the result DataFrame
    print(result_df)

    return result_df

    # # Extract and organize values into arrays
    # parameters = ['mean', 'variance', 'contrast', 'dissimilarity', 'homogeneity', 
    #               'asm', 'entropy', 'maximum', 'correlation']
    # bands = ['band1', 'band2', 'band3', 'band4']
    
    # data = {}
    # # Inside the final loop
    # for parameter in parameters:
    #     for stat in ['mean', 'std', 'skewness', 'kurtosis', 'CV']:
    #         for band_index, band in enumerate(bands, start=1):
    #             key = f"{parameter}_{stat}_{band}"
    #             data[key] = np.array([d[key] for d in data_list])
                

                

    # # Create a DataFrame
    # df = pd.DataFrame(data)
    # print(df)

    return df



# In[ ]:


df1=fynbos_texture(raster_path1, raster_filename1, shapefiles1)

#%% alternately

#load raster1
def fynbos_texture(raster_path, raster_filename, shapefiles):
    raster = rxr.open_rasterio(Path(raster_path, raster_filename),masked=True).squeeze()
    geometries = shapefiles.geometry.apply(mapping)
    
    
    # Create an empty list to store the clipped rasters
    clipped_rasters = []
    #for i in range(len(shapefiles1)):
    for i in range(4):    
        # Clip the raster with the current geometry
        p1 = raster.rio.clip([geometries[i]],shapefiles.crs)
        # Convert the clipped raster to a NumPy array
        p1_np = np.asarray(p1)
        # Append the clipped raster array to the list
        clipped_rasters.append(p1_np)
        
    import sys
    from texture import fastglcm_wrapper
    sys.path.append('/S:/mc5545/SA_Drone_data')
    from scipy.stats import skew, kurtosis
    
    # Initialize an empty list to store results for all rasters
    data_list = []
    # Loop through each raster
    for idx, raster in enumerate(clipped_rasters):

        # clipped_rasters is of shape (4, 553, 552) 
        #The order is  G, R, RE, NIR for our stack
        raster = np.transpose(raster, (1, 2, 0))
        # Now, clipped_rasters will be of shape (553, 552, 4) with the order of the bands preserved
        raster.shape

        # Create an instance of the fastglcm_wrapper class
        # Specify the parameters: levels, kernel_size, distance_offset, and angles
        tex2 = fastglcm_wrapper(raster, levels=8, kernel_size=5, distance_offset=5, angles=[0, 45, 90, 135])
       
        # Calculate various GLCM matrices
        mean = tex2.calculate_glcm_mean()
        variance =tex2.calculate_glcm_var()
        contrast = tex2.calculate_glcm_contrast()
        dissimilarity = tex2.calculate_glcm_dissimilarity()
        homogeneity = tex2.calculate_glcm_homogenity()
        asm = tex2.calculate_glcm_asm()
        entropy = tex2.calculate_glcm_entropy()
        maximum = tex2.calculate_glcm_max()
        correlation = tex2.calculate_glcm_correlation()
        

        # # Calculate the means and standard deviations of each channel
        # def calculate_stats(tex_features):
        #     mean_tex_features = np.nanmean(tex_features, axis=(0, 1)).reshape(1, -1)
        #     std_tex_features = np.nanstd(tex_features, axis=(0, 1)).reshape(1, -1)
        #     skewness_tex_features = skew(tex_features, axis=(0, 1), nan_policy='omit').reshape(1, -1)
        #     kurtosis_tex_features = kurtosis(tex_features, axis=(0, 1), nan_policy='omit').reshape(1, -1)
        #     CV_tex_features = (std_tex_features / mean_tex_features)
        #     # Return results in a dictionary
        #     return (mean_tex_features, std_tex_features,skewness_tex_features, 
        #              kurtosis_tex_features, CV_tex_features)
    
        # Calculate the means and standard deviations of each channel
        def calculate_stats(tex_features):
           mean_tex_features = np.nanmean(tex_features, axis=(0, 1))
           std_tex_features = np.nanstd(tex_features, axis=(0, 1))
           skewness_tex_features = skew(tex_features, axis=(0, 1), nan_policy='omit')
           kurtosis_tex_features = kurtosis(tex_features, axis=(0, 1), nan_policy='omit')
           CV_tex_features = (std_tex_features / mean_tex_features)
           # Return results in a dictionary
           return (mean_tex_features, std_tex_features,skewness_tex_features, 
                    kurtosis_tex_features, CV_tex_features)


        mean_,std_mean,skewness_mean,kurtosis_mean,CV_mean=calculate_stats(mean)

        # Append the results for the current raster to the list
        data_list.append({
          'mean': mean_,
          'std_mean': std_mean,
          'skewness_mean': skewness_mean,
          'kurtosis_mean': kurtosis_mean,
          'CV_mean': CV_mean
          })

    
    # Extract and organize values into arrays
    cv_mean_band1 = np.array([d['CV_mean'][0] for d in data_list])
    cv_mean_band2 = np.array([d['CV_mean'][1] for d in data_list])
    cv_mean_band3 = np.array([d['CV_mean'][2] for d in data_list])
    cv_mean_band4 = np.array([d['CV_mean'][3] for d in data_list])

    kurtosis_mean_band1 = np.array([d['kurtosis_mean'][0] for d in data_list])
    kurtosis_mean_band2 = np.array([d['kurtosis_mean'][1] for d in data_list])
    kurtosis_mean_band3 = np.array([d['kurtosis_mean'][2] for d in data_list])
    kurtosis_mean_band4 = np.array([d['kurtosis_mean'][3] for d in data_list])

    mean_band1 = np.array([d['mean'][0] for d in data_list])
    mean_band2 = np.array([d['mean'][1] for d in data_list])
    mean_band3 = np.array([d['mean'][2] for d in data_list])
    mean_band4 = np.array([d['mean'][3] for d in data_list])

    skewness_mean_band1 = np.array([d['skewness_mean'][0] for d in data_list])
    skewness_mean_band2 = np.array([d['skewness_mean'][1] for d in data_list])
    skewness_mean_band3 = np.array([d['skewness_mean'][2] for d in data_list])
    skewness_mean_band4 = np.array([d['skewness_mean'][3] for d in data_list])

    std_mean_band1 = np.array([d['std_mean'][0] for d in data_list])
    std_mean_band2 = np.array([d['std_mean'][1] for d in data_list])
    std_mean_band3 = np.array([d['std_mean'][2] for d in data_list])
    std_mean_band4 = np.array([d['std_mean'][3] for d in data_list])

    # Create a DataFrame
    df = pd.DataFrame({
        'CV_mean_band1': cv_mean_band1,
        'CV_mean_band2': cv_mean_band2,
        'CV_mean_band3': cv_mean_band3,
        'CV_mean_band4': cv_mean_band4,
        'kurtosis_mean_band1': kurtosis_mean_band1,
        'kurtosis_mean_band2': kurtosis_mean_band2,
        'kurtosis_mean_band3': kurtosis_mean_band3,
        'kurtosis_mean_band4': kurtosis_mean_band4,
        'mean_band1': mean_band1,
        'mean_band2': mean_band2,
        'mean_band3': mean_band3,
        'mean_band4': mean_band4,
        'skewness_mean_band1': skewness_mean_band1,
        'skewness_mean_band2': skewness_mean_band2,
        'skewness_mean_band3': skewness_mean_band3,
        'skewness_mean_band4': skewness_mean_band4,
        'std_mean_band1': std_mean_band1,
        'std_mean_band2': std_mean_band2,
        'std_mean_band3': std_mean_band3,
        'std_mean_band4': std_mean_band4
        })

    print(df)

    

    return
#%%
