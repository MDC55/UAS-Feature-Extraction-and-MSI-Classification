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

# In[7]:


#stacking Green, Red, Red Edge, and Near Infrared band individual rasters
#into one multi-band raster
#top_path='/gdrive/My Drive/Fynbos/October_2023/Grootbos_Drone_fil/s/mavic3m/'
top_path='S:/mc5545/SA_Drone_data/'
raster_path = top_path+'burnplot18_lr/reflectance'
raster_filename = 'burnplot18_lr_band_stack.tif'
#load shapefile
#shp = gpd.read_file(Path(top_path,'burnplot18_lr/burnplot18_lr_shape_all/burnplot18_lr_shp.shp'))
shp = gpd.read_file(Path(top_path,'burnplot18_lr/burnplot18_lr_shape/burnplot18_lr_shp.shp'))
#load raster
raster = rxr.open_rasterio(Path(raster_path, raster_filename),masked=True).squeeze()

geometries = shp.geometry.apply(mapping)


# In[12]:
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


# In[46]:


from pathlib import Path
# the path of the directory which saves your .py file
src_dir = Path('/content/drive/MyDrive/Colab Notebooks/')

# add the path to system path
import sys
try:
  sys.path.index(str(src_dir))
except ValueError:
  sys.path.insert(0,str(src_dir))

# print system path
sys.path


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

plt.imshow((p1_np_n[:,:,3]))

    
# In[78]:


# Assuming p1_np is your array
np.isnan(p1_np)
np.sum(np.isnan(p1_np))


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


# In[ ]:


mean[:,:,1].shape


# In[51]:


# Create subplots for all the GLCM matrices
fig, axes = plt.subplots(3, 3, figsize=(12, 12))

i=3

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
fig, axes = plt.subplots(3, 3, figsize=(12, 12))

cax_mean = axes[0, 0].imshow(mean[:, :, i], cmap='viridis')
axes[0, 0].set_title('Mean GLCM')
fig.colorbar(cax_mean, ax=axes[0, 0], fraction=0.046, pad=0.04)
axes[0, 0].axis('off')

cax_variance = axes[0, 1].imshow(variance[:, :, i], cmap='viridis')
axes[0, 1].set_title('Variance GLCM')
fig.colorbar(cax_variance, ax=axes[0, 1], fraction=0.046, pad=0.04)
axes[0, 1].axis('off')

cax_contrast = axes[0, 2].imshow(contrast[:, :, i], cmap='viridis')
axes[0, 2].set_title('Contrast GLCM')
fig.colorbar(cax_contrast, ax=axes[0, 2], fraction=0.046, pad=0.04)
axes[0, 2].axis('off')


cax_contrast = axes[1, 0].imshow(dissimilarity[:, :, i], cmap='viridis')
axes[1, 0].set_title('Dissimilarity GLCM')
fig.colorbar(cax_contrast, ax=axes[1, 0], fraction=0.046, pad=0.04)
axes[1, 0].axis('off')

cax_contrast = axes[1, 1].imshow(homogeneity[:, :, i], cmap='viridis')
axes[1, 1].set_title('Homogeneity GLCM')
fig.colorbar(cax_contrast, ax=axes[1, 1], fraction=0.046, pad=0.04)
axes[1, 1].axis('off')

cax_contrast = axes[1, 2].imshow(asm[:, :, i], cmap='viridis')
axes[1, 2].set_title('ASM GLCM')
fig.colorbar(cax_contrast, ax=axes[1, 2], fraction=0.046, pad=0.04)
axes[1, 2].axis('off')


cax_contrast = axes[2, 0].imshow(entropy[:, :, i], cmap='viridis')
axes[2, 0].set_title('Entropy GLCM')
fig.colorbar(cax_contrast, ax=axes[2, 0], fraction=0.046, pad=0.04)
axes[2, 0].axis('off')

cax_contrast = axes[2, 1].imshow(maximum[:, :, i], cmap='viridis')
axes[2, 1].set_title('Maximum GLCM')
fig.colorbar(cax_contrast, ax=axes[2, 1], fraction=0.046, pad=0.04)
axes[2, 1].axis('off')

cax_contrast = axes[2, 2].imshow(correlation[:, :, i], cmap='viridis')
axes[2, 2].set_title('Correlation GLCM')
fig.colorbar(cax_contrast, ax=axes[2, 2], fraction=0.046, pad=0.04)
axes[2, 2].axis('off')

# Adjust the layout
plt.tight_layout()

# Display the subplots
plt.show()

#%%

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
