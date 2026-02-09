#!/usr/bin/env python
# coding: utf-8

# In[5]:


#importing packages
import os
os.environ['USE_PYGEOS'] = '0'
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Load indices DataFrame from the CSV file that has spectral and texture features for truth value

#df1  #2006
#df2  #2016
#df3  #2016
#df4  #2019
#df5  #2020
directory_path='D:/Drive F 5-11-2024/2023 SA Fynbos Field Work/Drone data analysis code/from 2022 trip/Indices_csv/'

#directory_path='S:/mc5545/SA_Drone_data/from 2022 trip/Indices_csv/'
df1_i = pd.read_csv(f'{directory_path}indices_2006.csv', index_col=0)
df2_i = pd.read_csv(f'{directory_path}indices_2016_1.csv', index_col=0)
df3_i = pd.read_csv(f'{directory_path}indices_2016_2.csv', index_col=0)
df4_i = pd.read_csv(f'{directory_path}indices_2019.csv', index_col=0)
df5_i = pd.read_csv(f'{directory_path}indices_2020.csv', index_col=0)
#df6_i = pd.read_csv(f'{directory_path}indices_2022.csv', index_col=0)

# Load textures DataFrame from the CSV file
directory_path='D:/Drive F 5-11-2024/2023 SA Fynbos Field Work/Drone data analysis code/from 2022 trip/textures_csv/'
#directory_path='S:/mc5545/SA_Drone_data/from 2022 trip/textures_csv/'
df1_t = pd.read_csv(f'{directory_path}textures_2006.csv', index_col=0)
df2_t = pd.read_csv(f'{directory_path}textures_2016_1.csv', index_col=0)
df3_t = pd.read_csv(f'{directory_path}textures_2016_2.csv', index_col=0)
df4_t = pd.read_csv(f'{directory_path}textures_2019.csv', index_col=0)
df5_t = pd.read_csv(f'{directory_path}textures_2020.csv', index_col=0)
#df6_t = pd.read_csv(f'{directory_path}textures_2022.csv', index_col=0)
# Concatenate the indices and textures features into a new df along the columns (axis=1) 
df1 = pd.concat([df1_i, df1_t], axis=1) #2006
df2 = pd.concat([df2_i, df2_t], axis=1) #2016
df3 = pd.concat([df3_i, df3_t], axis=1) #2016
df4 = pd.concat([df4_i, df4_t], axis=1) #2019
df5 = pd.concat([df5_i, df5_t], axis=1) #2020
#df6 = pd.concat([df6_i, df6_t], axis=1) #2022

df1['Alpha_diversity'] = 2.0345
df2['Alpha_diversity'] = 2.0761
df3['Alpha_diversity'] = 2.0281
df4['Alpha_diversity'] = 2.1669 #
df5['Alpha_diversity'] = 2.2425
#df6['Alpha_diversity'] = 2.2959

#dataframes = [df2, df3]
# Concatenate DataFrames
#concatenated_df = pd.concat(dataframes)

del (df1_i,df2_i,df3_i,df4_i,df5_i,df1_t,df2_t,df3_t,df4_t,df5_t)

columns_to_keep = ['mean_NIR', 'CV_RedEdge', 'CVI', 'skewness_DSWI4', 'kurtosis_PSRI',
                   'M3Cl', 'CV_LCI', 'CV_ratio1', 'dissimilarity_mean_band2',
                   'homogeneity_mean_band4','Alpha_diversity','x','y']

# Assuming df1, df2, df3, df4, and df5 are your dataframes
df1 = df1[columns_to_keep]
df2 = df2[columns_to_keep]
df3 = df3[columns_to_keep]
df4 = df4[columns_to_keep]
df5 = df5[columns_to_keep]



''' Load indices DataFrame from the CSV file that has spectral and texture features 
and which alpha diversity to be determined value
'''

#Without alpha diversity data
# Load indices DataFrame from the CSV file
directory_path='D:/Drive F 5-11-2024/2023 SA Fynbos Field Work/Drone data analysis code/Indices_csv/'
#directory_path='S:/mc5545/SA_Drone_data/Indices_csv/'
df1_i = pd.read_csv(f'{directory_path}indices_2006_burnplot18.csv', index_col=0)
df2_i = pd.read_csv(f'{directory_path}indices_2016_burn2016.csv', index_col=0)
df3_i = pd.read_csv(f'{directory_path}indices_2017_burn2017.csv', index_col=0)
df4_i = pd.read_csv(f'{directory_path}indices_2019_burn2019.csv', index_col=0)
df5_i = pd.read_csv(f'{directory_path}indices_2020_burnplot17.csv', index_col=0)
df6_i = pd.read_csv(f'{directory_path}indices_2022_burn2022.csv', index_col=0)


# Load textures DataFrame from the CSV file
directory_path='D:/Drive F 5-11-2024/2023 SA Fynbos Field Work/Drone data analysis code/textures_csv/'
#directory_path='S:/mc5545/SA_Drone_data/textures_csv/'
df1_t = pd.read_csv(f'{directory_path}textures_2006_burnplot18.csv', index_col=0)
df2_t = pd.read_csv(f'{directory_path}textures_2016_burn2016.csv', index_col=0)
df3_t = pd.read_csv(f'{directory_path}textures_2017_burn2017.csv', index_col=0)
df4_t = pd.read_csv(f'{directory_path}textures_2019_burn2019.csv', index_col=0)
df5_t = pd.read_csv(f'{directory_path}textures_2020_burnplot17.csv', index_col=0)
df6_t = pd.read_csv(f'{directory_path}textures_2022_burn2022.csv', index_col=0)

# Concatenate the indices and textures features into a new df along the columns (axis=1) 
df1_all = pd.concat([df1_i, df1_t], axis=1)
df2_all = pd.concat([df2_i, df2_t], axis=1)
df3_all = pd.concat([df3_i, df3_t], axis=1)
df4_all = pd.concat([df4_i, df4_t], axis=1)
df5_all = pd.concat([df5_i, df5_t], axis=1)
df6_all = pd.concat([df6_i, df6_t], axis=1)


del (df1_i,df2_i,df3_i,df4_i,df5_i,df6_i,
     df1_t,df2_t,df3_t,df4_t,df5_t,df6_t)



del(df3_all,df6_all)

columns_to_keep = ['mean_NIR', 'CV_RedEdge', 'CVI', 'skewness_DSWI4', 'kurtosis_PSRI',
                   'M3Cl', 'CV_LCI', 'CV_ratio1', 'dissimilarity_mean_band2',
                   'homogeneity_mean_band4','x','y']

# Assuming df1, df2, df3, df4, and df5 are your dataframes
df1_all = df1_all[columns_to_keep]
df2_all = df2_all[columns_to_keep]
df4_all = df4_all[columns_to_keep]
df5_all = df5_all[columns_to_keep]

#%%
#Fine Tuning

from scipy.spatial.distance import euclidean

def estimate_alpha_diversity(df_all, df_known_alpha_diversity):
    # Extract features and coordinates from the dataframes
    features_all = df_all.drop(['x', 'y'], axis=1).values
    coordinates_all = df_all[['x', 'y']].values
    
    coordinates_known = df_all[['x', 'y']].values
    
    known_features = df_known_alpha_diversity.drop(['x', 'y','Alpha_diversity'], axis=1).values
    known_alpha_diversity = df_known_alpha_diversity['Alpha_diversity'].values[0]  # Assuming only one known alpha diversity

    # Calculate similarity metric (Euclidean distance) between features and coordinates
    distances = np.array([euclidean(features_all[i], known_features[0]) 
                          #+ euclidean(coordinates_all[i], coordinates_known[0]) 
                          for i in range(len(features_all))])
    
    # Compute weights based on inverse distances
    weights = 1 / (1 + distances)
    mean_weight = np.mean(weights)
    # Weighted average of alpha diversities
    estimated_alpha_diversity = np.sum(weights * known_alpha_diversity) / np.sum(weights)
    
    # Calculate individual alpha diversities for each coordinate
    individual_alpha_diversities = (weights*known_alpha_diversity)/mean_weight

    
    # Calculate individual alpha diversities for each coordinate
    #individual_alpha_diversities = weights * known_alpha_diversity

    return individual_alpha_diversities,estimated_alpha_diversity

# Apply the approach to each dataframe and create separate heatmap plots
dfs_all = [df1_all, df2_all, df4_all, df5_all]
known_alpha_diversity_dfs = [df1, df2, df4, df5]



years = [2006, 2016, 2019, 2020]

for i, (df_all, known_alpha_diversity_df, year) in enumerate(zip(dfs_all, known_alpha_diversity_dfs, years)):
    individual_alpha_diversities, estimated_alpha_diversity = estimate_alpha_diversity(df_all, known_alpha_diversity_df)

    # Reshape individual alpha diversities into a 2D array
    side_length = int(np.sqrt(len(individual_alpha_diversities)))
    alpha_diversities_matrix = individual_alpha_diversities[:side_length**2].reshape((side_length, side_length))

    # Create a heatmap
    #plt.imshow(alpha_diversities_matrix, cmap='viridis')
    
    # Create a heatmap with fixed color scale
    plt.imshow(
    alpha_diversities_matrix, 
    cmap='viridis',
    vmin=0.65,
    vmax=4.57
     )
    #plt.title(f'Year: {year} | Estimated Alpha Diversity: {estimated_alpha_diversity:.4f}')
    plt.title(f'Year: {year}')
    #plt.xlabel('X-coordinate')
    #plt.ylabel('Y-coordinate')
    # OPTIONAL: also hide axis tick labels
    plt.xticks([])
    plt.yticks([])

    #plt.colorbar(label='Individual Alpha Diversity')
    #cbar = plt.colorbar(label='Individual Alpha Diversity')
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=12, width=1.5)
    for tick in cbar.ax.get_yticklabels():
        tick.set_weight('bold')
    
    # Increase font size of colorbar label
    #cbar.ax.yaxis.label.set_fontsize(16)  # Adjust the fontsize as needed
    
    # Increase font size of colorbar tick labels
    #cbar.ax.tick_params(axis='y', labelsize=16)  # Adjust the fontsize as needed
    plt.savefig('D:/Drive F 5-11-2024/2023 SA Fynbos Field Work/1. Writing-Project 2/Study Area Fynbos/alpha2020.png',dpi=300)
    plt.show()
    plt.show()

dfx=  df1.drop(['Alpha_diversity'], axis=1)  
estimated_alpha_2006,all_sum=estimate_alpha_diversity(df1_all, df1)
estimated_alpha_2006_,all_sum=estimate_alpha_diversity(dfx, df1)
estimated_alpha_2016,all_sum=estimate_alpha_diversity(df2_all, df2)
estimated_alpha_2019,all_sum=estimate_alpha_diversity(df4_all, df4)  
estimated_alpha_2020,all_sum=estimate_alpha_diversity(df5_all, df5)


#%%
import numpy as np

# Calculate statistics for estimated_alpha_2006
min_alpha_2006 = np.min(estimated_alpha_2006)
max_alpha_2006 = np.max(estimated_alpha_2006)
mean_alpha_2006 = np.mean(estimated_alpha_2006)
std_alpha_2006 = np.std(estimated_alpha_2006)

# Calculate statistics for estimated_alpha_2016
min_alpha_2016 = np.min(estimated_alpha_2016)
max_alpha_2016 = np.max(estimated_alpha_2016)
mean_alpha_2016 = np.mean(estimated_alpha_2016)
std_alpha_2016 = np.std(estimated_alpha_2016)

# Calculate statistics for estimated_alpha_2019
min_alpha_2019 = np.min(estimated_alpha_2019)
max_alpha_2019 = np.max(estimated_alpha_2019)
mean_alpha_2019 = np.mean(estimated_alpha_2019)
std_alpha_2019 = np.std(estimated_alpha_2019)

# Calculate statistics for estimated_alpha_2020
min_alpha_2020 = np.min(estimated_alpha_2020)
max_alpha_2020 = np.max(estimated_alpha_2020)
mean_alpha_2020 = np.mean(estimated_alpha_2020)
std_alpha_2020 = np.std(estimated_alpha_2020)

# Print the results
print("Statistics for estimated_alpha_2006:")
print("Minimum:", min_alpha_2006)
print("Maximum:", max_alpha_2006)
print("Mean:", mean_alpha_2006)
print("Standard Deviation:", std_alpha_2006)
print()

print("Statistics for estimated_alpha_2016:")
print("Minimum:", min_alpha_2016)
print("Maximum:", max_alpha_2016)
print("Mean:", mean_alpha_2016)
print("Standard Deviation:", std_alpha_2016)
print()

print("Statistics for estimated_alpha_2019:")
print("Minimum:", min_alpha_2019)
print("Maximum:", max_alpha_2019)
print("Mean:", mean_alpha_2019)
print("Standard Deviation:", std_alpha_2019)
print()

print("Statistics for estimated_alpha_2020:")
print("Minimum:", min_alpha_2020)
print("Maximum:", max_alpha_2020)
print("Mean:", mean_alpha_2020)
print("Standard Deviation:", std_alpha_2020)

#%%
columns_to_keep = ['mean_NIR', 'CV_RedEdge', 'CVI', 'skewness_DSWI4', 'kurtosis_PSRI',
                   'M3Cl', 'CV_LCI', 'CV_ratio1', 'dissimilarity_mean_band2',
                   'homogeneity_mean_band4','x','y']

# Assuming df1, df2, df3, df4, and df5 are your dataframes
df1_all = df1_all[columns_to_keep]
estimated_alpha_2006,all_sum=estimate_alpha_diversity(df1_all, df1)


#%%

print(len(estimated_alpha_2006))
print(len(estimated_alpha_2016))
print(len(estimated_alpha_2019))
print(len(estimated_alpha_2020))




# Create separate dataframes for each array
data_2006 = pd.DataFrame({'Estimated Alpha': estimated_alpha_2006})
data_2016 = pd.DataFrame({'Estimated Alpha': estimated_alpha_2016})
data_2019 = pd.DataFrame({'Estimated Alpha': estimated_alpha_2019})
data_2020 = pd.DataFrame({'Estimated Alpha': estimated_alpha_2020})

# Concatenate the dataframes with a new 'Year' column
data_2006['Year'] = 'Year 2006'
data_2016['Year'] = 'Year 2016'
data_2019['Year'] = 'Year 2019'
data_2020['Year'] = 'Year 2020'


# Concatenate all dataframes into one
df = pd.concat([data_2006, data_2016, data_2019, data_2020], ignore_index=True)

# Setting the theme and palette
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0), 'axes.linewidth':2})
palette = sns.color_palette("RdBu", 10) #pastel #Set2

# Create a FacetGrid with row-wise Year category
g = sns.FacetGrid(df, palette=palette, row="Year", hue="Year", aspect=7, height=1.2)

# Map KDE plots for each Year
g.map_dataframe(sns.kdeplot, x="Estimated Alpha", fill=True, alpha=1)
g.map_dataframe(sns.kdeplot, x="Estimated Alpha", color='black')

# Function to label each plot
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, color='black', fontsize=20, ha="left", va="center", 
            transform=ax.transAxes)

# Add labels to each plot
g.map(label, "Year")
g.set_titles("")
# Adjust subplot spacing
g.fig.subplots_adjust(hspace=-.5)
# Remove yticks, set common xlabel, and remove left spine
g.set(yticks=[], ylabel="",xlabel="Estimated Alpha")

# Set y-axis label fontsize and weight
for ax in g.axes.flat:
    ax.set_xlabel("Shannon Weiner Index", fontsize=20, weight='bold') 
    ax.tick_params(axis='x', labelsize=20)  # Set x-axis tick label size
    #ax.set_ylabel("Estimated Alpha", fontsize=20, weight='bold') 
    ax.tick_params(axis='y', labelsize=20)  # Set y-axis tick label size

# Set x-axis tick labels to bold
for ax in g.axes.flat:
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')

g.despine(left=True)

# Set title
plt.suptitle('Ridge Plot of Estimated Alpha Diversity by Year', y=0.98,fontsize=20, 
             fontweight='bold')
# Save the plot in a directory with 200 DPI resolution
#plt.savefig('F:/2023 SA Fynbos Field Work/1. Writing-Project 2/Study Area Fynbos/ridgeplot.png', dpi=300)
#plt.savefig('S:/mc5545/SA_Drone_data/study_areas/ridgeplot.png', dpi=300)

plt.show()


Alpha_2006= 2.0345
Alpha_2016_1 = 2.0761
Alpha_2016_2 = 2.0281
Alpha_2019 = 2.1669

Alpha_2020= 2.2425 


#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Create separate dataframes for each array
# data_2006 = pd.DataFrame({'Estimated Alpha': estimated_alpha_2006[1]})
# data_2016 = pd.DataFrame({'Estimated Alpha': estimated_alpha_2016[1]})
# data_2019 = pd.DataFrame({'Estimated Alpha': estimated_alpha_2019[1]})
# data_2020 = pd.DataFrame({'Estimated Alpha': estimated_alpha_2020[1]})

# Create separate dataframes for each array
data_2006 = pd.DataFrame({'Estimated Alpha': estimated_alpha_2006})
data_2016 = pd.DataFrame({'Estimated Alpha': estimated_alpha_2016})
data_2019 = pd.DataFrame({'Estimated Alpha': estimated_alpha_2019})
data_2020 = pd.DataFrame({'Estimated Alpha': estimated_alpha_2020})

# Concatenate the dataframes with a new 'Year' column
data_2006['Year'] = 'Year 2006'
data_2016['Year'] = 'Year 2016'
data_2019['Year'] = 'Year 2019'
data_2020['Year'] = 'Year 2020'


# Concatenate all dataframes into one
df = pd.concat([data_2006, data_2016, data_2019, data_2020], ignore_index=True)

# Setting the theme and palette
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0), 'axes.linewidth':2})
palette = sns.color_palette("RdBu", 10)  # pastel #Set2

# Create a FacetGrid with row-wise Year category
g = sns.FacetGrid(df, palette=palette, row="Year", hue="Year", aspect=7, height=1.2)

# Map KDE plots for each Year
g.map_dataframe(sns.kdeplot, x="Estimated Alpha", fill=True, alpha=1)
g.map_dataframe(sns.kdeplot, x="Estimated Alpha", color='black')

# Function to label each plot
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, color='black', fontsize=20, ha="left", va="center", 
            transform=ax.transAxes)

# Add labels to each plot
g.map(label, "Year")
g.set_titles("")
# Adjust subplot spacing
g.fig.subplots_adjust(hspace=-.5)
# Remove yticks, set common xlabel, and remove left spine
g.set(yticks=[], ylabel="", xlabel="Estimated Alpha")

# Set y-axis label fontsize and weight
for ax in g.axes.flat:
    ax.set_xlabel("Shannon Weiner Index", fontsize=20, weight='bold') 
    ax.tick_params(axis='x', labelsize=20)  # Set x-axis tick label size
    ax.tick_params(axis='y', labelsize=20)  # Set y-axis tick label size

# Set x-axis tick labels to bold
for ax in g.axes.flat:
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')

g.despine(left=True)

# Add vertical lines for alpha values of each year
alpha_values = {
    'Year 2006': 2.0345,
    'Year 2016': 2.0761,
    'Year 2019': 2.1669, #
    'Year 2020': 2.2425
}

for ax, (year, alpha_value) in zip(g.axes.flat, alpha_values.items()):
    ax.axvline(alpha_value, color='red', linestyle='--')

# Set title
plt.suptitle('Ridge Plot of Estimated Alpha Diversity by Year', y=0.98, fontsize=20, fontweight='bold')
# Save the plot in a directory with 200 DPI resolution
#plt.savefig('F:/2023 SA Fynbos Field Work/1. Writing-Project 2/Study Area Fynbos/ridgeplot1.png', dpi=300)

plt.show()


#%%
#importing packages
import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features


#stacked Green, Red, Red Edge, and Near Infrared band individual rasters into one multi-band raster
#op_path='S:/mc5545/SA_Drone_data/'
top_path='D:/Drive F 5-11-2024/2023 SA Fynbos Field Work/Drone data analysis code/'
shapefiles_2006 = gpd.read_file(Path(top_path,'burnplot18_lr/burnplot18_lr_shape_all/burnplot18_lr_shp.shp'))
shapefiles_2016 = gpd.read_file(Path(top_path,'burn2016_lr/burn2016_lr_shape_all/burn2016_shp.shp'))
#shapefiles_2017 = gpd.read_file(Path(top_path,'burn2017&2016_lr/burn2017_shape/burn2017_shp.shp'))
shapefiles_2019 = gpd.read_file(Path(top_path,'burn2019_lr/burn2019_lr_shape_all/burn2019_lr_shp.shp'))
shapefiles_2020 = gpd.read_file(Path(top_path,'burnplot17_lr/burnplot17_lr_shape_all/burnplot17_lr_shp.shp'))
#shapefiles_2022 = gpd.read_file(Path(top_path,'burn2022_lr/burn2022_lr_shape_all/burn2022_lr_shp.shp'))



# Add the alpha values to the shapefile's attribute table
shapefiles_2006['alpha'] = estimated_alpha_2006
shapefiles_2016['alpha'] = estimated_alpha_2016
shapefiles_2019['alpha'] = estimated_alpha_2019
shapefiles_2020['alpha'] = estimated_alpha_2020

# Save the updated shapefile
#shapefiles_2006.to_file('S:/mc5545/SA_Drone_data/burnplot18_lr/alpha_2006/shapefile_2006.shp')
#shapefiles_2016.to_file('S:/mc5545/SA_Drone_data/burn2016_lr/alpha_2016/shapefile_2016.shp')
#shapefiles_2019.to_file('S:/mc5545/SA_Drone_data/burn2019_lr/alpha_2019/shapefile_2019.shp')
#shapefiles_2020.to_file('S:/mc5545/SA_Drone_data/burnplot17_lr/alpha_2020/shapefile_2020.shp')

#import rioxarray as rxr

raster_path1 = top_path+'burnplot18_lr/reflectance/burnplot18_lr_band_stack.tif'

#x='S:/mc5545/SA_Drone_data/burnplot18_lr/reflectance/burnplot18_lowres_transparent_reflectance_red edge.tif'

#raster = rxr.open_rasterio(Path(raster_path1, raster_filename1),masked=True).squeeze()

raster_path2 = top_path+'burn2016_lr/reflectance/burn2016_lr_band_stack.tif'

raster_path4 = top_path+'burn2019_lr/reflectance/burn2019_lr_band_stack.tif'

raster_path5 = top_path+'burnplot17_lr/reflectance/burnplot17_lr_band_stack.tif'

#%%

# Define the metadata for the raster
with rasterio.open(raster_path1) as src:
  meta = src.meta
# Create a new raster file
with rasterio.open('D:/Drive F 5-11-2024/2023 SA Fynbos Field Work/Drone data analysis code/burnplot18_lr/alpha_2006_new/est_alpha_2006.tif', 
                    'w', **meta) as dst:
    # Convert the polygons to raster
    out_image = features.rasterize(
        [(geom, value) for geom, value in zip(shapefiles_2006.geometry, shapefiles_2006.alpha)],
        out_shape=(meta['height'], meta['width']),
        transform=dst.transform,
        fill=np.nan,
        all_touched=True,
        dtype='float64'
    )
    dst.write(out_image, 1)    

#%%
# Define the metadata for the raster
with rasterio.open(raster_path2) as src:
  meta = src.meta

# Create a new raster file
with rasterio.open('S:/mc5545/SA_Drone_data/burn2016_lr/alpha_2016 new/est_alpha_2016.tif', 
                    'w', **meta) as dst:
    # Convert the polygons to raster
    out_image = features.rasterize(
        [(geom, value) for geom, value in zip(shapefiles_2016.geometry, shapefiles_2016.alpha)],
        out_shape=(meta['height'], meta['width']),
        transform=dst.transform,
        fill=np.nan,
        all_touched=True,
        dtype='float64'
    )
    dst.write(out_image, 1)        
    
#%%
# Define the metadata for the raster
with rasterio.open(raster_path4) as src:
  meta = src.meta

# Create a new raster file
with rasterio.open('S:/mc5545/SA_Drone_data/burn2019_lr/alpha_2019_new/est_alpha_2019_man.tif', 
                    'w', **meta) as dst:

    # Convert the polygons to raster
    out_image = features.rasterize(
        [(geom, value) for geom, value in zip(shapefiles_2019.geometry, shapefiles_2019.alpha)],
        out_shape=(meta['height'], meta['width']),
        transform=dst.transform,
        fill=np.nan,
        all_touched=True,
        dtype='float64'
    )
    dst.write(out_image, 1)        
    
    
#%%

# Define the metadata for the raster
with rasterio.open(raster_path5) as src:
  meta = src.meta

# Create a new raster file
with rasterio.open('S:/mc5545/SA_Drone_data/burnplot17_lr/alpha_2020_new/est_alpha_2020.tif', 
                    'w', **meta) as dst:
    
    # Convert the polygons to raster
    out_image = features.rasterize(
        [(geom, value) for geom, value in zip(shapefiles_2020.geometry, shapefiles_2020.alpha)],
        out_shape=(meta['height'], meta['width']),
        transform=dst.transform,
        fill=np.nan,
        all_touched=True,
        dtype='float64'
    )
    dst.write(out_image, 1)        
    


    
#%%

import rasterio
import matplotlib.pyplot as plt

# Corrected file path with raw string literal to avoid issues with backslashes
file_path = r'D:\Drive F 5-11-2024\2023 SA Fynbos Field Work\Drone data analysis code\burn2016_lr\alpha_2016 new\est_alpha_2016.tif'

try:
    # Open the TIFF file using rasterio
    with rasterio.open(file_path) as image:
        # Read the first band of the image
        image_array = image.read(1)  # Read the first band; adjust if needed

    # Plot the image
    plt.imshow(image_array, cmap='viridis', vmin=1, vmax=4.26)
    plt.axis('off')  # Turn off axis
    plt.colorbar(label='Individual Alpha Diversity')
    plt.title('TIFF Image')
    plt.show()

except rasterio.errors.RasterioIOError:
    print(f"Could not open the file. Please check that the path '{file_path}' is correct and the file exists.")

#%% used it on AGU plot

# Create separate dataframes for each array
data_2006 = pd.DataFrame({'Estimated Alpha': estimated_alpha_2006})
data_2016 = pd.DataFrame({'Estimated Alpha': estimated_alpha_2016})
data_2019 = pd.DataFrame({'Estimated Alpha': estimated_alpha_2019})
data_2020 = pd.DataFrame({'Estimated Alpha': estimated_alpha_2020})

# Concatenate the dataframes with a new 'Year' column
data_2006['Year'] = 'Year 2006'
data_2016['Year'] = 'Year 2016'
data_2019['Year'] = 'Year 2019'
data_2020['Year'] = 'Year 2020'


# Concatenate all dataframes into one
df = pd.concat([data_2006, data_2016, data_2019, data_2020], ignore_index=True)

# Setting the theme and palette
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0), 'axes.linewidth':2})
palette = sns.color_palette("RdBu", 10) #pastel #Set2

# Create a FacetGrid with row-wise Year category
g = sns.FacetGrid(df, palette=palette, row="Year", hue="Year", aspect=7, height=1.2)

# Map KDE plots for each Year
g.map_dataframe(sns.kdeplot, x="Estimated Alpha", fill=True, alpha=1)
g.map_dataframe(sns.kdeplot, x="Estimated Alpha", color='black')

# Function to label each plot
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, color='black', fontsize=18, ha="left", va="center", 
            transform=ax.transAxes)

# Add labels to each plot
g.map(label, "Year")
g.set_titles("")
# Adjust subplot spacing
g.fig.subplots_adjust(hspace=-.5)
# Remove yticks, set common xlabel, and remove left spine
g.set(yticks=[], ylabel="",xlabel="Estimated Alpha")

# Set y-axis label fontsize and weight
for ax in g.axes.flat:
    #ax.set_xlabel("Shannon Weiner Index", fontsize=20, weight='bold') 
    ax.set_xlabel("Shannon Weiner Index", fontsize=18) 
    ax.tick_params(axis='x', labelsize=18)  # Set x-axis tick label size
    #ax.set_ylabel("Estimated Alpha", fontsize=20, weight='bold') 
    ax.tick_params(axis='y', labelsize=18)  # Set y-axis tick label size

# # Set x-axis tick labels to bold
# for ax in g.axes.flat:
#     for label in ax.get_xticklabels():
#         label.set_fontweight('bold')

g.despine(left=True)

# Set title
# plt.suptitle('Ridge Plot of Estimated Alpha Diversity by Year', y=0.98,fontsize=20, 
#              fontweight='bold')
plt.suptitle('Ridge Plot of Estimated Alpha Diversity by Year', y=0.98,fontsize=18)
# Save the plot in a directory with 200 DPI resolution
#plt.savefig('F:/2023 SA Fynbos Field Work/1. Writing-Project 2/Study Area Fynbos/ridgeplot.png', dpi=300)
#plt.savefig('S:/mc5545/SA_Drone_data/study_areas/ridgeplot.png', dpi=300)

plt.show()
