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

# Load indices DataFrame from the CSV file
directory_path='S:/mc5545/SA_Drone_data/from 2022 trip/Indices_csv/'
df1_i = pd.read_csv(f'{directory_path}indices_2006.csv', index_col=0)
df2_i = pd.read_csv(f'{directory_path}indices_2016_1.csv', index_col=0)
df3_i = pd.read_csv(f'{directory_path}indices_2016_2.csv', index_col=0)
df4_i = pd.read_csv(f'{directory_path}indices_2019.csv', index_col=0)
df5_i = pd.read_csv(f'{directory_path}indices_2020.csv', index_col=0)


# Load textures DataFrame from the CSV file
directory_path='S:/mc5545/SA_Drone_data/from 2022 trip/textures_csv/'
df1_t = pd.read_csv(f'{directory_path}textures_2006.csv', index_col=0)
df2_t = pd.read_csv(f'{directory_path}textures_2016_1.csv', index_col=0)
df3_t = pd.read_csv(f'{directory_path}textures_2016_2.csv', index_col=0)
df4_t = pd.read_csv(f'{directory_path}textures_2019.csv', index_col=0)
df5_t = pd.read_csv(f'{directory_path}textures_2020.csv', index_col=0)

# Concatenate the indices and textures features into a new df along the columns (axis=1) 
df1 = pd.concat([df1_i, df1_t], axis=1) #2006
df2 = pd.concat([df2_i, df2_t], axis=1) #2016
df3 = pd.concat([df3_i, df3_t], axis=1) #2016
df4 = pd.concat([df4_i, df4_t], axis=1) #2019
df5 = pd.concat([df5_i, df5_t], axis=1) #2020


df1['Alpha_diversity'] = 2.0345
df2['Alpha_diversity'] = 2.0761
df3['Alpha_diversity'] = 2.0281
df4['Alpha_diversity'] = 1.8669
df5['Alpha_diversity'] = 2.2425


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




#Without alpha diversity data
# Load indices DataFrame from the CSV file
directory_path='S:/mc5545/SA_Drone_data/Indices_csv/'
df1_i = pd.read_csv(f'{directory_path}indices_2006_burnplot18.csv', index_col=0)
df2_i = pd.read_csv(f'{directory_path}indices_2016_burn2016.csv', index_col=0)
df3_i = pd.read_csv(f'{directory_path}indices_2017_burn2017.csv', index_col=0)
df4_i = pd.read_csv(f'{directory_path}indices_2019_burn2019.csv', index_col=0)
df5_i = pd.read_csv(f'{directory_path}indices_2020_burnplot17.csv', index_col=0)
df6_i = pd.read_csv(f'{directory_path}indices_2022_burn2022.csv', index_col=0)


# Load textures DataFrame from the CSV file
directory_path='S:/mc5545/SA_Drone_data/textures_csv/'
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

#dataframes = [df1, df2, df3, df4, df5, df6]

# df1['Year'] = 2006
# df2['Year'] = 2016
# df3['Year'] = 2017
# df4['Year'] = 2019
# df5['Year'] = 2020
# df6['Year'] = 2022



# # Concatenate DataFrames
# concatenated_df = pd.concat(dataframes)
# # List of columns to drop
# columns_to_drop = ['x', 'y']

# # Drop the specified columns
# X= concatenated_df.drop(columns=columns_to_drop)
# X = X.drop("Year", axis=1)  # Assuming "year" is the column containing the target variable

# y = concatenated_df["Year"]

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


# In[ ]:


# # dataframes without Alpha_diversity column but has all the common features
# df1_all  #2006
# df2_all  #2016
# #df3_all  #2017
# df4_all  #2019
# df5_all  #2020
# #df6_all  #2022


# # dataframes with available Alpha_diversity column
# df1  #2006
# df2  #2016
# df3  #2016
# df4  #2019
# df5  #2020




# In[30]:

#https://towardsdatascience.com/17-types-of-similarity-and-dissimilarity-measures-used-in-data-science-3eb914d2681
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.spatial.distance import euclidean

# def estimate_alpha_diversity(df_all, df_known_alpha_diversity):
#     # Extract features and coordinates from the dataframes
#     features_all = df_all.drop(['x', 'y'], axis=1).values
#     coordinates_all = df_all[['x', 'y']].values
    
#     coordinates_known = df_all[['x', 'y']].values
    
#     known_features = df_known_alpha_diversity.drop(['x', 'y','Alpha_diversity'], axis=1).values
#     known_alpha_diversity = df_known_alpha_diversity['Alpha_diversity'].values[0]  # Assuming only one known alpha diversity

#     # Calculate similarity metric (Euclidean distance) between features and coordinates
#     distances = np.array([euclidean(features_all[i], known_features[0]) 
#                           #+ euclidean(coordinates_all[i], coordinates_known[0]) 
#                           for i in range(len(features_all))])
    
#     # Calculate similarity metric (Euclidean distance) between features and coordinates
# #     distances = np.array([euclidean(features_all[i], features_all[0]) + 
# #                           euclidean(coordinates_all[i], coordinates_known[0]) 
# #                           for i in range(len(features_all))])
    
# #     # Calculate similarity metric (Euclidean distance) between features and coordinates
# #     distances = np.array([euclidean(features_all[i], known_features[0]) + 
# #                           euclidean(coordinates_all[i], coordinates_all[0]) 
# #                           for i in range(len(features_all))])

# #     # Calculate similarity metric (Euclidean distance) between features and coordinates
# #     distances = np.array([euclidean(features_all[i], features_all[0]) + 
# #                           euclidean(coordinates_all[i], coordinates_all[0]) 
# #                           for i in range(len(features_all))])
    
#     # Compute weights based on inverse distances
#     weights = 1 / (1 + distances)

#     # Weighted average of alpha diversities
#     estimated_alpha_diversity = np.sum(weights * known_alpha_diversity) / np.sum(weights)
    
#     # Calculate individual alpha diversities for each coordinate
#     individual_alpha_diversities = weights * known_alpha_diversity

#     return individual_alpha_diversities,estimated_alpha_diversity

# # Apply the approach to each dataframe and create separate heatmap plots
# dfs_all = [df1_all, df2_all, df4_all, df5_all]
# known_alpha_diversity_dfs = [df1, df2, df4, df5]

# for i, (df_all, known_alpha_diversity_df) in enumerate(zip(dfs_all, known_alpha_diversity_dfs)):
#     individual_alpha_diversities = estimate_alpha_diversity(df_all, known_alpha_diversity_df)

#     # Reshape individual alpha diversities into a 2D array
#     side_length = int(np.sqrt(len(individual_alpha_diversities)))
#     alpha_diversities_matrix = individual_alpha_diversities[:side_length**2].reshape((side_length, side_length))

#     # Create a heatmap
#     plt.imshow(alpha_diversities_matrix, cmap='viridis')
#     plt.title(f'Dataframe {i + 1}')
#     plt.xlabel('X-coordinate')
#     plt.ylabel('Y-coordinate')
#     plt.colorbar(label='Individual Alpha Diversity')
#     plt.show()


# for i, df_all in enumerate(dfs_all):
#     estimated_alpha_diversity = estimate_alpha_diversity(df_all, known_alpha_diversity_dfs[i])
#     print(f"Estimated Alpha Diversity for {i+1}: {estimated_alpha_diversity:.4f}")

#%%
#Fine Tuning

#
#df1  #2006
#df2  #2016
#df3  #2016
#df4  #2019
#df5  #2020

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

    # Weighted average of alpha diversities
    estimated_alpha_diversity = np.sum(weights * known_alpha_diversity) / np.sum(weights)
    
    # Calculate individual alpha diversities for each coordinate
    individual_alpha_diversities = weights * known_alpha_diversity

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
    plt.imshow(alpha_diversities_matrix, cmap='viridis')
    #plt.title(f'Year: {year} | Estimated Alpha Diversity: {estimated_alpha_diversity:.4f}')
    plt.title(f'Year: {year}')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.colorbar(label='Individual Alpha Diversity')
    
    plt.show()
    
estimated_alpha_2006,all_sum=estimate_alpha_diversity(df1_all, df1)
estimated_alpha_2016,all_sum=estimate_alpha_diversity(df2_all, df2)
estimated_alpha_2019,all_sum=estimate_alpha_diversity(df4_all, df4)  
estimated_alpha_2020,all_sum=estimate_alpha_diversity(df5_all, df5)

#%%
import matplotlib.pyplot as plt

# Plot histogram
plt.figure(figsize=(10, 5))
#plt.hist(estimated_alpha_2006, bins=40, color='blue', alpha=0.5,rwidth=0.8,edgecolor='black')
plt.hist(estimated_alpha_2006,bins=40,color='skyblue',edgecolor='black')
plt.title('Burn Year 2006', fontsize=18, fontweight='bold')  # Increase font size and font weight of title
plt.xlabel('Shannon-Weiner Index', fontsize=14, fontweight='bold')  # Increase font size and font weight of x-axis label
plt.ylabel('No of Field Quadarts', fontsize=14, fontweight='bold')  # Increase font size and font weight of y-axis label
plt.xticks(fontsize=14,fontweight='bold')  # Increase font size of x-axis tick labels
plt.yticks(fontsize=14,fontweight='bold')  # Increase font size of y-axis tick labels
plt.grid(True)
# Save the plot in a directory with 200 DPI resolution
plt.savefig('S:/mc5545/SA_Drone_data/study_areas/alpha2006_hist.png', dpi=200)

plt.show()


#%%
import matplotlib.pyplot as plt

# Plot histogram
plt.figure(figsize=(10, 5))
#plt.hist(estimated_alpha_2006, bins=40, color='blue', alpha=0.5,rwidth=0.8,edgecolor='black')
plt.hist(estimated_alpha_2016,bins=40,color='skyblue',edgecolor='black')
plt.title('Burn Year 2016', fontsize=18, fontweight='bold')  # Increase font size and font weight of title
plt.xlabel('Shannon-Weiner Index', fontsize=14, fontweight='bold')  # Increase font size and font weight of x-axis label
plt.ylabel('No of Field Quadarts', fontsize=14, fontweight='bold')  # Increase font size and font weight of y-axis label
plt.xticks(fontsize=14,fontweight='bold')  # Increase font size of x-axis tick labels
plt.yticks(fontsize=14,fontweight='bold')  # Increase font size of y-axis tick labels
plt.grid(True)
# Save the plot in a directory with 200 DPI resolution
plt.savefig('S:/mc5545/SA_Drone_data/study_areas/alpha2016_hist.png', dpi=200)

plt.show()


#%%
import matplotlib.pyplot as plt

# Plot histogram
plt.figure(figsize=(10, 5))
#plt.hist(estimated_alpha_2006, bins=40, color='blue', alpha=0.5,rwidth=0.8,edgecolor='black')
plt.hist(estimated_alpha_2019,bins=40,color='skyblue',edgecolor='black')
plt.title('Burn Year 2019', fontsize=18, fontweight='bold')  # Increase font size and font weight of title
plt.xlabel('Shannon-Weiner Index', fontsize=14, fontweight='bold')  # Increase font size and font weight of x-axis label
plt.ylabel('No of Field Quadarts', fontsize=14, fontweight='bold')  # Increase font size and font weight of y-axis label
plt.xticks(fontsize=14,fontweight='bold')  # Increase font size of x-axis tick labels
plt.yticks(fontsize=14,fontweight='bold')  # Increase font size of y-axis tick labels
plt.grid(True)
# Save the plot in a directory with 200 DPI resolution
plt.savefig('S:/mc5545/SA_Drone_data/study_areas/alpha2019_hist.png', dpi=200)

plt.show()

#%%
import matplotlib.pyplot as plt

# Plot histogram
plt.figure(figsize=(10, 5))
#plt.hist(estimated_alpha_2006, bins=40, color='blue', alpha=0.5,rwidth=0.8,edgecolor='black')
plt.hist(estimated_alpha_2020,bins=40,color='skyblue',edgecolor='black')
plt.title('Burn Year 2020', fontsize=18, fontweight='bold')  # Increase font size and font weight of title
plt.xlabel('Shannon-Weiner Index', fontsize=14, fontweight='bold')  # Increase font size and font weight of x-axis label
plt.ylabel('No of Field Quadarts', fontsize=14, fontweight='bold')  # Increase font size and font weight of y-axis label
plt.xticks(fontsize=14,fontweight='bold')  # Increase font size of x-axis tick labels
plt.yticks(fontsize=14,fontweight='bold')  # Increase font size of y-axis tick labels
plt.grid(True)
# Save the plot in a directory with 200 DPI resolution
plt.savefig('S:/mc5545/SA_Drone_data/study_areas/alpha2020_hist.png', dpi=200)

plt.show()

#%%

import warnings



import seaborn as sns
import matplotlib.pyplot as plt

# Combine the data into a single list of samples
my_samples = [
    estimated_alpha_2006,
    estimated_alpha_2016,
    estimated_alpha_2019,
    estimated_alpha_2020
]

# Create a seaborn ridge plot
sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

# Create the ridge plot
sns.kdeplot(data=my_samples, shade=True)

# Add labels and title
plt.xlabel('Estimated Alpha')
plt.ylabel('Density')
plt.title('Ridge Plot of Estimated Alpha')

# Show the plot
plt.show()

#%%

import seaborn as sns
import matplotlib.pyplot as plt

# Create separate DataFrames for each year
df_2006 = pd.DataFrame({'Estimated Alpha': estimated_alpha_2006})
df_2016 = pd.DataFrame({'Estimated Alpha': estimated_alpha_2016})
df_2019 = pd.DataFrame({'Estimated Alpha': estimated_alpha_2019})
df_2020 = pd.DataFrame({'Estimated Alpha': estimated_alpha_2020})

# Create FacetGrid for each year
g_2006 = sns.FacetGrid(data=df_2006, aspect=9, height=1.2)
g_2016 = sns.FacetGrid(data=df_2016, aspect=9, height=1.2)
g_2019 = sns.FacetGrid(data=df_2019, aspect=9, height=1.2)
g_2020 = sns.FacetGrid(data=df_2020, aspect=9, height=1.2)

# Plot kernel density estimate with fill for each year
g_2006.map_dataframe(sns.kdeplot, x='Estimated Alpha', fill=True, alpha=1)
g_2016.map_dataframe(sns.kdeplot, x='Estimated Alpha', fill=True, alpha=1)
g_2019.map_dataframe(sns.kdeplot, x='Estimated Alpha', fill=True, alpha=1)
g_2020.map_dataframe(sns.kdeplot, x='Estimated Alpha', fill=True, alpha=1)

# Plot kernel density estimate without fill (outline) for each year
g_2006.map_dataframe(sns.kdeplot, x='Estimated Alpha', color='black')
g_2016.map_dataframe(sns.kdeplot, x='Estimated Alpha', color='black')
g_2019.map_dataframe(sns.kdeplot, x='Estimated Alpha', color='black')
g_2020.map_dataframe(sns.kdeplot, x='Estimated Alpha', color='black')

# Adjust subplot spacing for each year
g_2006.fig.subplots_adjust(hspace=-.5)
g_2016.fig.subplots_adjust(hspace=-.5)
g_2019.fig.subplots_adjust(hspace=-.5)
g_2020.fig.subplots_adjust(hspace=-.5)

# Remove yticks, set xlabel for each year
g_2006.set(yticks=[], xlabel="Estimated Alpha")
g_2016.set(yticks=[], xlabel="Estimated Alpha")
g_2019.set(yticks=[], xlabel="Estimated Alpha")
g_2020.set(yticks=[], xlabel="Estimated Alpha")

# Remove left spine for each year
g_2006.despine(left=True)
g_2016.despine(left=True)
g_2019.despine(left=True)
g_2020.despine(left=True)

# Add title for each year
g_2006.fig.suptitle('Ridge Plot of Estimated Alpha - 2006', y=0.98)
g_2016.fig.suptitle('Ridge Plot of Estimated Alpha - 2016', y=0.98)
g_2019.fig.suptitle('Ridge Plot of Estimated Alpha - 2019', y=0.98)
g_2020.fig.suptitle('Ridge Plot of Estimated Alpha - 2020', y=0.98)

# Show the plots
plt.show()


#%%

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Create a DataFrame with the estimated alpha values for each burn year
data = pd.DataFrame({
    'Burn Year 2006': estimated_alpha_2006,
    'Burn Year 2016': estimated_alpha_2016,
    'Burn Year 2019': estimated_alpha_2019,
    'Burn Year 2020': estimated_alpha_2020
})

# Create ridge plot
plt.figure(figsize=(10, 6))
for column in data.columns:
    sns.kdeplot(data[column], fill=True, linewidth=0.5, alpha=0.5)
    plt.plot([], label=column)


# Set labels and title
plt.xlabel('Shannon-Weiner Index')
plt.ylabel('Density')
plt.title('Ridge Plot of Shannon-Weiner Index for Different Burn Years')
plt.legend()
# Show plot
plt.show()


#%%%

import numpy as np
from ridgeplot import ridgeplot
import matplotlib.pyplot as plt

# Combine the data into a single list of samples
my_samples = [
    estimated_alpha_2006,
    estimated_alpha_2016,
    estimated_alpha_2019,
    estimated_alpha_2020
]

# Plot the ridge plot
fig = ridgeplot(samples=my_samples)

# Update layout if needed
fig.update_layout(height=450, width=800)

# Show the plot
fig.show()


#%%
print(len(estimated_alpha_2006))
print(len(estimated_alpha_2016))
print(len(estimated_alpha_2019))
print(len(estimated_alpha_2020))


#%%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Generate sample data for each burn year (replace with your actual data)
estimated_alpha_2006 = np.random.normal(loc=10, scale=3, size=100)
estimated_alpha_2016 = np.random.normal(loc=15, scale=2, size=100)
estimated_alpha_2019 = np.random.normal(loc=20, scale=4, size=100)
estimated_alpha_2020 = np.random.normal(loc=25, scale=1, size=100)

# Create a new figure and specify that it's a 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot 3D histograms for each burn year
bins = 30  # Adjust as needed

hist, bins = np.histogram(estimated_alpha_2006, bins=bins)
ax.bar(bins[:-1], hist, zs=0, zdir='y', alpha=0.5, label='Burn Year 2006')

hist, bins = np.histogram(estimated_alpha_2016, bins=bins)
ax.bar(bins[:-1], hist, zs=1, zdir='y', alpha=0.5, label='Burn Year 2016')

hist, bins = np.histogram(estimated_alpha_2019, bins=bins)
ax.bar(bins[:-1], hist, zs=2, zdir='y', alpha=0.5, label='Burn Year 2019')

hist, bins = np.histogram(estimated_alpha_2020, bins=bins)
ax.bar(bins[:-1], hist, zs=3, zdir='y', alpha=0.5, label='Burn Year 2020')

# Set labels and title
ax.set_xlabel('Shannon-Weiner Index')
ax.set_ylabel('Burn Year')
ax.set_zlabel('Frequency')
ax.set_title('3D Histogram of Shannon-Weiner Index for Different Burn Years')

# Add legend
ax.legend()

# Show plot
plt.show()

#%%
#importing packages
import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features


#stacked Green, Red, Red Edge, and Near Infrared band individual rasters into one multi-band raster
top_path='S:/mc5545/SA_Drone_data/'
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
with rasterio.open('S:/mc5545/SA_Drone_data/burnplot18_lr/alpha_2006/est_alpha_2006.tif', 
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
with rasterio.open('S:/mc5545/SA_Drone_data/burn2016_lr/alpha_2016/est_alpha_2016.tif', 
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
with rasterio.open('S:/mc5545/SA_Drone_data/burn2019_lr/alpha_2019/est_alpha_2019.tif', 
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
with rasterio.open('S:/mc5545/SA_Drone_data/burnplot17_lr/alpha_2020/est_alpha_2020.tif', 
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

