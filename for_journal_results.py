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

#S:\mc5545\SA_Drone_data\burnplot18_lr\for_texture_j
top_path='S:/mc5545/SA_Drone_data/'
#shapefiles1 = gpd.read_file(Path(top_path,'burnplot18_lr/for_texture_j/tx2006n.shp'))
shapefiles1 = gpd.read_file(Path(top_path,'burnplot18_lr/burnplot18_lr_shape_all/burnplot18_lr_shp.shp'))
# shapefiles2 = gpd.read_file(Path(top_path,'burn2016_lr/burn2016_lr_shape_all/burn2016_shp.shp'))
# shapefiles3 = gpd.read_file(Path(top_path,'burn2017&2016_lr/burn2017_shape/burn2017_shp.shp'))
# shapefiles4 = gpd.read_file(Path(top_path,'burn2019_lr/burn2019_lr_shape_all/burn2019_lr_shp.shp'))
# shapefiles5 = gpd.read_file(Path(top_path,'burnplot17_lr/burnplot17_lr_shape_all/burnplot17_lr_shp.shp'))
# shapefiles6 = gpd.read_file(Path(top_path,'burn2022_lr/burn2022_lr_shape_all/burn2022_lr_shp.shp'))


raster_path1 = top_path+'burnplot18_lr/reflectance'
raster_filename1 = 'burnplot18_lr_band_stack.tif'

# raster_path2 = top_path+'burn2016_lr/reflectance'
# raster_filename2 = 'burn2016_lr_band_stack.tif'

# raster_path3 = top_path+'burn2017&2016_lr/reflectance'
# raster_filename3 = 'burn2017&2016_lr_band_stack.tif'

# raster_path4 = top_path+'burn2019_lr/reflectance'
# raster_filename4 = 'burn2019_lr_band_stack.tif'

# raster_path5 = top_path+'burnplot17_lr/reflectance'
# raster_filename5 = 'burnplot17_lr_band_stack.tif'

# raster_path6 = top_path+'burn2022_lr/reflectance'
# raster_filename6 = 'burn2022_lr_band_stack.tif'





#load raster1
def raster_clip(raster_path, raster_filename, shapefiles):
    #stacking Green, Red, Red Edge, and Near Infrared band individual rasters
    #into one multi-band raster
    raster = rxr.open_rasterio(Path(raster_path, raster_filename),masked=True).squeeze()
    geometries = shapefiles.geometry.apply(mapping)
    # Transpose the array to have dimensions ('y', 'x', 'band')
    p1_np_n = np.transpose(raster.values, (1, 2, 0))
    
    # # Create an empty list to store the clipped rasters
    # clipped_rasters = []
    # for i in range(len(shapefiles)):
    # #for i in range(10):
    #     # Clip the raster with the current geometry
    #     p1 = raster.rio.clip([geometries[i]],shapefiles.crs)
    #     # Convert the clipped raster to a NumPy array
    #     p1_np = np.asarray(p1)
    #     # Append the clipped raster array to the list
    #     clipped_rasters.append(p1_np)
        
    return   p1_np_n  

#%%
df1=raster_clip(raster_path1, raster_filename1, shapefiles1)
# df2=raster_clip(raster_path2, raster_filename2, shapefiles2) 
# df3=raster_clip(raster_path3, raster_filename3, shapefiles3)
# df4=raster_clip(raster_path4, raster_filename4, shapefiles4)
# df5=raster_clip(raster_path5, raster_filename5, shapefiles5)
# df6=raster_clip(raster_path6, raster_filename6, shapefiles6)

#%%
df1g=df1[:,:,3]

#%%
import sys
sys.path.append('/S:/mc5545/SA_Drone_data')

# In[ ]:

from texture import fastglcm_wrapper

# Create an instance of the fastglcm_wrapper class
# Specify the parameters: levels, kernel_size, distance_offset, and angles
tex2 = fastglcm_wrapper(df1, levels=8, kernel_size=5, distance_offset=5, 
                        angles=[0, 45, 90, 135])

correlation = tex2.calculate_glcm_correlation()

#%%


# In[12]:
    
    
#stacking Green, Red, Red Edge, and Near Infrared band individual rasters
#into one multi-band raster
#top_path='/gdrive/My Drive/Fynbos/October_2023/Grootbos_Drone_fil/s/mavic3m/'
top_path='S:/mc5545/SA_Drone_data/'
raster_path = top_path+'burnplot18_lr/reflectance'
raster_filename = 'burnplot18_lr_band_stack.tif'
#load raster
raster = rxr.open_rasterio(Path(raster_path, raster_filename),masked=True).squeeze()


geometries = shapefiles1.geometry.apply(mapping)

#%%
import rasterio
import rasterio.plot
import xarray as xr
from pathlib import Path

# Assuming you have a 4-band raster with bands in the order: Green, Red, Red Edge, Near Infrared
green_band = raster.sel(band=1).values
red_band = raster.sel(band=2).values
red_edge_band = raster.sel(band=3).values
nir_band = raster.sel(band=4).values

# Calculate NDVI
ndvi = (nir_band - red_band) / (nir_band + red_band)

# Save the NDVI raster
ndvi_filename = 'burnplot18_lr_ndvi.tif'

# Assuming you have the original metadata from the raster
ndvi_meta = rasterio.open(Path(raster_path, raster_filename)).meta
ndvi_meta.update(dtype=rasterio.float32, count=1)

with rasterio.open(Path(raster_path, ndvi_filename), 'w', **ndvi_meta) as dst:
    dst.write(ndvi.astype(rasterio.float32), 1)

import matplotlib.pyplot as plt

# Plot NDVI
plt.imshow(ndvi, cmap='RdYlGn')  # Adjust the colormap and limits as needed
#plt.imshow(ndvi, cmap='RdYlGn', vmin=-0, vmax=1)
plt.colorbar(label='NDVI')
plt.axis('off')
plt.title('NDVI (Burn Year-2006)')
plt.show()

#%%

#clipped plot for first 10
# Create an empty list to store the clipped rasters
clipped_rasters = []

#for i in range(len(shapefiles1)):
for i in range(10):         
  # Clip the raster with the current geometry
  p1 = raster.rio.clip([geometries[i]],shapefiles1.crs)

  # Convert the clipped raster to a NumPy array
  p1_np = np.asarray(p1)
  #p1_np = np.nan_to_num(p1_np, nan=0)
  # Append the clipped raster array to the list
  clipped_rasters.append(p1_np)


# In[ ]:


# Assuming you have already created the `clipped_rasters` list as shown in the previous response.
  #so the order is now G=0, R=1, RE=2, NIR=3 for our stack
# Create an empty list to store the NDVI values for each clipped raster
ReCI_values = []
NDRE_values = []
GNDVI_values =[]
OSAVI_values= []
GCI_values  = []
SR_values   = []
MSR_values  = []
RDVI_values=  []

# Iterate through the clipped rasters
for p1_np in clipped_rasters:
    # Calculate NDVI using the formula
    ReCI = (p1_np[3] / p1_np[2])-1
    NDRE=  (p1_np[3]-p1_np[2])/(p1_np[3]+p1_np[2])
    GNDVI= (p1_np[3]-p1_np[0])/(p1_np[3]+p1_np[0])
    OSAVI= (p1_np[3]-p1_np[1])/(p1_np[3]+p1_np[1]+0.16)
    GCI=   (p1_np[3]/p1_np[0])-1
    SR=    (p1_np[3] / p1_np[1])
    MSR =  ((p1_np[3] / p1_np[1])-1)/(np.sqrt((p1_np[3] / p1_np[1])+1))
    RDVI=  np.sqrt((p1_np[3]-p1_np[1])/(p1_np[3]+p1_np[1]))

    # Append the NDVI value to the list
    ReCI_values.append(ReCI)
    NDRE_values.append(NDRE)
    GNDVI_values.append(GNDVI)
    OSAVI_values.append(OSAVI)
    GCI_values.append(GCI)
    SR_values.append(SR)
    MSR_values.apend (MSR)
    RDVI_values.append(RDVI)
# Now, the `ndvi_values` list contains the NDVI values for all the clipped rasters.




# In[55]:


# p1_np is of shape (4, 553, 552)
#The order is  G, R, RE, NIR for our stack
p1_np
p1_np_n = np.transpose(p1_np, (1, 2, 0))
#p1_np[3]
# Now, p1_np will be of shape (553, 552, 4) with the order of the bands preserved
p1_np_n.shape


# In[ ]:


p1_np_n[:,:,3]

# In[78]:

plt.imshow((p1_np_n[:,:,2]))

    
# In[78]:


# Assuming p1_np is your array
np.isnan(p1_np[3])
np.sum(np.isnan(p1_np[3]))


# In[ ]:


np.isnan(p1_np)


# In[ ]:
import sys
sys.path.append('/S:/mc5545/SA_Drone_data')

# In[ ]:

from texture import fastglcm_wrapper

# Create an instance of the fastglcm_wrapper class
# Specify the parameters: levels, kernel_size, distance_offset, and angles
tex2 = fastglcm_wrapper(p1_np_n, levels=8, kernel_size=5, distance_offset=5, angles=[0, 45, 90, 135])


# In[69]:


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




# In[51]:


# Create subplots for all the GLCM matrices
fig, axes = plt.subplots(3, 3, figsize=(12, 12))

i=2

# Plot GLCM matrices
axes[0, 0].imshow(mean[:,:,i], cmap='viridis')
axes[0, 0].set_title('Mean GLCM')
axes[0, 1].imshow(variance[:,:,i], cmap='viridis')
axes[0, 1].set_title('Variance GLCM')
axes[0, 2].imshow(contrast [:,:,i], cmap='viridis')
axes[0, 2].set_title('Contrast GLCM')

axes[1, 0].imshow(dissimilarity [:,:,i], cmap='viridis')
axes[1, 0].set_title('Dissimilarity GLCM')
axes[1, 1].imshow(homogeneity [:,:,i], cmap='viridis')
axes[1, 1].set_title('Homogeneity GLCM')
axes[1, 2].imshow(asm [:,:,i], cmap='viridis')
axes[1, 2].set_title('ASM GLCM')

axes[2, 0].imshow(entropy [:,:,i], cmap='viridis')
axes[2, 0].set_title('Entropy GLCM')
axes[2, 1].imshow(maximum [:,:,i], cmap='viridis')
axes[2, 1].set_title('Maximum GLCM')
axes[2, 2].imshow(correlation [:,:,i], cmap='terrain')
axes[2, 2].set_title('Correlation GLCM')

# Adjust the layout
plt.tight_layout()

# Display the subplots
plt.show()

#%%
# Create subplots for all the GLCM matrices
fig, axes = plt.subplots(3, 4, figsize=(12, 8))
cmap='terrain'
#cmap='viridis'
i=2
cax_mean = axes[0, 0].imshow(p1_np_n[:, :, i]) #, cmap='jet'
axes[0, 0].set_title('Red-edge Band ')
fig.colorbar(cax_mean, ax=axes[0, 0], fraction=0.046, pad=0.04)
axes[0, 0].axis('off')

cax_mean = axes[0, 1].imshow(mean[:, :, i], cmap=cmap)
axes[0, 1].set_title('Mean GLCM')
fig.colorbar(cax_mean, ax=axes[0, 1], fraction=0.046, pad=0.04)
axes[0, 1].axis('off')

cax_variance = axes[0, 2].imshow(variance[:, :, i], cmap=cmap)
axes[0, 2].set_title('Variance GLCM')
fig.colorbar(cax_variance, ax=axes[0, 2], fraction=0.046, pad=0.04)
axes[0, 2].axis('off')

cax_contrast = axes[0, 3].imshow(contrast[:, :, i], cmap=cmap)
axes[0, 3].set_title('Contrast GLCM')
fig.colorbar(cax_contrast, ax=axes[0, 3], fraction=0.046, pad=0.04)
axes[0, 3].axis('off')


cax_contrast = axes[1, 0].imshow(dissimilarity[:, :, i], cmap=cmap)
axes[1, 0].set_title('Dissimilarity GLCM')
fig.colorbar(cax_contrast, ax=axes[1, 0], fraction=0.046, pad=0.04)
axes[1, 0].axis('off')

cax_contrast = axes[1, 1].imshow(homogeneity[:, :, i], cmap=cmap)
axes[1, 1].set_title('Homogeneity GLCM')
fig.colorbar(cax_contrast, ax=axes[1, 1], fraction=0.046, pad=0.04)
axes[1, 1].axis('off')

cax_contrast = axes[1, 2].imshow(asm[:, :, i], cmap=cmap)
axes[1, 2].set_title('ASM GLCM')
fig.colorbar(cax_contrast, ax=axes[1, 2], fraction=0.046, pad=0.04)
axes[1, 2].axis('off')


cax_contrast = axes[1, 3].imshow(entropy[:, :, i], cmap=cmap)
axes[1, 3].set_title('Entropy GLCM')
fig.colorbar(cax_contrast, ax=axes[1, 3], fraction=0.046, pad=0.04)
axes[1, 3].axis('off')

cax_contrast = axes[2, 0].imshow(maximum[:, :, i], cmap=cmap)
axes[2, 0].set_title('Maximum GLCM')
fig.colorbar(cax_contrast, ax=axes[2, 0], fraction=0.046, pad=0.04)
axes[2, 0].axis('off')

cax_contrast = axes[2, 1].imshow(correlation[:, :, i], cmap=cmap)
axes[2, 1].set_title('Correlation GLCM')
fig.colorbar(cax_contrast, ax=axes[2, 1], fraction=0.046, pad=0.04)
axes[2, 1].axis('off')

# Adjust the layout
plt.tight_layout()

# Specify the folder path and save the figure with 200 DPI
folder_path = 'S:/mc5545/SA_Drone_data/study_areas'
file_name = 'texture_map.png'
file_path = f'{folder_path}/{file_name}'
plt.savefig(file_path, dpi=200)

# Display the subplots
plt.show()

#%%

# Increase fontsize and use 'terrain' colormap
fontsize = 18
fontweight = 'bold'
cmap = 'terrain'

# Create subplots for all the GLCM matrices
fig, axes = plt.subplots(3, 3, figsize=(16,12))

cax_mean = axes[0, 0].imshow(mean[:, :, i], cmap=cmap)
mean_colorbar = fig.colorbar(cax_mean, ax=axes[0, 0], fraction=0.046, pad=0.04)
mean_colorbar.ax.tick_params(labelsize=fontsize)
axes[0, 0].set_title('Mean GLCM', fontsize=fontsize, fontweight=fontweight)
axes[0, 0].axis('off')

cax_variance = axes[0, 1].imshow(variance[:, :, i], cmap=cmap)
variance_colorbar = fig.colorbar(cax_variance, ax=axes[0, 1], fraction=0.046, pad=0.04)
variance_colorbar.ax.tick_params(labelsize=fontsize)
axes[0, 1].set_title('Variance GLCM', fontsize=fontsize, fontweight=fontweight)
axes[0, 1].axis('off')

cax_contrast = axes[0, 2].imshow(contrast[:, :, i], cmap=cmap)
contrast_colorbar = fig.colorbar(cax_contrast, ax=axes[0, 2], fraction=0.046, pad=0.04)
contrast_colorbar.ax.tick_params(labelsize=fontsize)
axes[0, 2].set_title('Contrast GLCM', fontsize=fontsize, fontweight=fontweight)
axes[0, 2].axis('off')

cax_contrast = axes[1, 0].imshow(dissimilarity[:, :, i], cmap=cmap)
dissimilarity_colorbar = fig.colorbar(cax_contrast, ax=axes[1, 0], fraction=0.046, pad=0.04)
dissimilarity_colorbar.ax.tick_params(labelsize=fontsize)
axes[1, 0].set_title('Dissimilarity GLCM', fontsize=fontsize, fontweight=fontweight)
axes[1, 0].axis('off')

cax_contrast = axes[1, 1].imshow(homogeneity[:, :, i], cmap=cmap)
homogeneity_colorbar = fig.colorbar(cax_contrast, ax=axes[1, 1], fraction=0.046, pad=0.04)
homogeneity_colorbar.ax.tick_params(labelsize=fontsize)
axes[1, 1].set_title('Homogeneity GLCM', fontsize=fontsize, fontweight=fontweight)
axes[1, 1].axis('off')

cax_contrast = axes[1, 2].imshow(asm[:, :, i], cmap=cmap)
asm_colorbar = fig.colorbar(cax_contrast, ax=axes[1, 2], fraction=0.046, pad=0.04)
asm_colorbar.ax.tick_params(labelsize=fontsize)
axes[1, 2].set_title('ASM GLCM', fontsize=fontsize, fontweight=fontweight)
axes[1, 2].axis('off')

cax_contrast = axes[2, 0].imshow(entropy[:, :, i], cmap=cmap)
entropy_colorbar = fig.colorbar(cax_contrast, ax=axes[2, 0], fraction=0.046, pad=0.04)
entropy_colorbar.ax.tick_params(labelsize=fontsize)
axes[2, 0].set_title('Entropy GLCM', fontsize=fontsize, fontweight=fontweight)
axes[2, 0].axis('off')

cax_contrast = axes[2, 1].imshow(maximum[:, :, i], cmap=cmap)
maximum_colorbar = fig.colorbar(cax_contrast, ax=axes[2, 1], fraction=0.046, pad=0.04)
maximum_colorbar.ax.tick_params(labelsize=fontsize)
axes[2, 1].set_title('Maximum GLCM', fontsize=fontsize, fontweight=fontweight)
axes[2, 1].axis('off')

cax_contrast = axes[2, 2].imshow(correlation[:, :, i], cmap=cmap)
correlation_colorbar = fig.colorbar(cax_contrast, ax=axes[2, 2], fraction=0.046, pad=0.04)
correlation_colorbar.ax.tick_params(labelsize=fontsize)
axes[2, 2].set_title('Correlation GLCM', fontsize=fontsize, fontweight=fontweight)
axes[2, 2].axis('off')

# Adjust the layout
plt.tight_layout()

# Specify the folder path and save the figure with 200 DPI
folder_path = 'S:/mc5545/SA_Drone_data/study_areas'
file_name = 'texture_map.png'
file_path = f'{folder_path}/{file_name}'
plt.savefig(file_path, dpi=200)

# Display the subplots
plt.show()

#%%


#%% Hints on Recovery
import matplotlib.pyplot as plt

# Data for 2006
total_pixels_2006 = 4464593
pixel_counts_2006 = [0, 12, 86247, 3048054, 1256061]

# Data for 2022
total_pixels_2022 = 4889598
pixel_counts_2022 = [447, 97682, 1769081, 2602675, 419713]

# Calculate percentages
percentages_2006 = [count / total_pixels_2006 * 100 for count in pixel_counts_2006]
percentages_2022 = [count / total_pixels_2022 * 100 for count in pixel_counts_2022]

# Plotting
classes = [0, 1, 2, 3, 4]
bar_width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

bar_2006 = ax.bar(classes, percentages_2006, width=bar_width, color='blue', alpha=0.7, label='2006')
bar_2022 = ax.bar([c + bar_width for c in classes], percentages_2022, width=bar_width, color='orange', alpha=0.7, label='2022')

# Add percentage labels on top of the bars
for bars, percentages in zip([bar_2006, bar_2022], [percentages_2006, percentages_2022]):
    for bar, percentage in zip(bars, percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{percentage:.2f}%', ha='center', va='bottom')

ax.set_xlabel('Class')
ax.set_ylabel('Percentage')
ax.set_title('Class Percentage Comparison (2006 vs 2022)')
ax.set_xticks([c + bar_width / 2 for c in classes])
# Edit here: Replace numeric class labels with class names
class_names = {0: 'Non Vegetation',  1: 'Low',2: 'Moderately Low',3: 'Moderately High',4: 'High' }
ax.set_xticklabels([class_names[c] for c in classes])

ax.legend()

plt.show()

#%%

from joblib import load
import os
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_sequential_feature_selection

# Load the trained Random Forest classifier
model_filepath = 'S:/mc5545/SA_Drone_data/trained_classifiers/random_forest_model.joblib'
random_forest_classifier = load(model_filepath)


#%% Classiiers summary
import matplotlib.pyplot as plt

# Data (same as before)
classifiers = [
    'KNN', 'NB', 'LR',
    'SVM', 'DT', 'RF',
    'GB', 'MLP'
]
test_accuracy = [96, 95, 97, 97, 92, 96, 96, 96]
kappa_coefficient = [94.55, 94.03, 96.14, 96.84, 89.98, 95.08, 95.08, 95.60]
precision = [95.70, 95.35, 96.96, 97.47, 92.13, 96.10, 96.09, 96.46]
recall = [95.65, 95.22, 96.91, 97.47, 91.99, 96.07, 96.07, 96.49]
f1_score = [95.59, 95.22, 96.92, 97.47, 91.97, 96.06, 96.04, 96.47]

# Create subplots
fig, ax = plt.subplots(figsize=(10, 6))

# Bar width
bar_width = 0.15

# Positions for bars
x = range(len(classifiers))

# Plot bars
ax.bar(x, test_accuracy, width=bar_width, label='Test Accuracy')
ax.bar([i + bar_width for i in x], kappa_coefficient, width=bar_width, label="Cohen's Kappa")
ax.bar([i + 2 * bar_width for i in x], precision, width=bar_width, label='Precision')
ax.bar([i + 3 * bar_width for i in x], recall, width=bar_width, label='Recall')
ax.bar([i + 4 * bar_width for i in x], f1_score, width=bar_width, label='F1 Score')

# Customize plot
ax.set_xticks([i + 2 * bar_width for i in x])
ax.set_xticklabels(classifiers, fontsize=16, fontweight='bold')  # Increase font size and weight
ax.set_xlabel('Classifiers', fontsize=16, fontweight='bold')  # Increase font size and weight
ax.set_ylabel('Performance Metrics Value', fontsize=16, fontweight='bold')  # Increase font size and weight
ax.set_title('Performance Metrics for Machine Learning Classifiers', fontsize=16, fontweight='bold')  # Increase font size and weight
ax.legend(loc='lower right', fontsize=16)  # Reposition the legend and increase font size

# Set y-axis limit
ax.set_ylim(80, 100)  # Adjust the limits as needed

# Increase y-axis tick font size and make them bold
ax.yaxis.set_tick_params(labelsize=16)
for label in ax.yaxis.get_ticklabels():
    label.set_weight('bold')

# Enable grid
ax.grid(True)

# Show plot
plt.tight_layout()
plt.show()
