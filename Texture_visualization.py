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

# #S:\mc5545\SA_Drone_data\burnplot18_lr\for_texture_j
top_path='S:/mc5545/SA_Drone_data/'
# #shapefiles1 = gpd.read_file(Path(top_path,'burnplot18_lr/for_texture_j/tx2006n.shp'))
shapefiles1 = gpd.read_file(Path(top_path,'burnplot18_lr/burnplot18_lr_shape_all/burnplot18_lr_shp.shp'))
# # shapefiles2 = gpd.read_file(Path(top_path,'burn2016_lr/burn2016_lr_shape_all/burn2016_shp.shp'))
# # shapefiles3 = gpd.read_file(Path(top_path,'burn2017&2016_lr/burn2017_shape/burn2017_shp.shp'))
# # shapefiles4 = gpd.read_file(Path(top_path,'burn2019_lr/burn2019_lr_shape_all/burn2019_lr_shp.shp'))
# # shapefiles5 = gpd.read_file(Path(top_path,'burnplot17_lr/burnplot17_lr_shape_all/burnplot17_lr_shp.shp'))
# # shapefiles6 = gpd.read_file(Path(top_path,'burn2022_lr/burn2022_lr_shape_all/burn2022_lr_shp.shp'))


# raster_path1 = top_path+'burnplot18_lr/reflectance'
# raster_filename1 = 'burnplot18_lr_band_stack.tif'

# # raster_path2 = top_path+'burn2016_lr/reflectance'
# # raster_filename2 = 'burn2016_lr_band_stack.tif'

# # raster_path3 = top_path+'burn2017&2016_lr/reflectance'
# # raster_filename3 = 'burn2017&2016_lr_band_stack.tif'

# # raster_path4 = top_path+'burn2019_lr/reflectance'
# # raster_filename4 = 'burn2019_lr_band_stack.tif'

# # raster_path5 = top_path+'burnplot17_lr/reflectance'
# # raster_filename5 = 'burnplot17_lr_band_stack.tif'

# # raster_path6 = top_path+'burn2022_lr/reflectance'
# # raster_filename6 = 'burn2022_lr_band_stack.tif'





# #load raster1
# def raster_clip(raster_path, raster_filename, shapefiles):
#     #stacking Green, Red, Red Edge, and Near Infrared band individual rasters
#     #into one multi-band raster
#     raster = rxr.open_rasterio(Path(raster_path, raster_filename),masked=True).squeeze()
#     geometries = shapefiles.geometry.apply(mapping)
#     # Transpose the array to have dimensions ('y', 'x', 'band')
#     p1_np_n = np.transpose(raster.values, (1, 2, 0))
    
#     # # Create an empty list to store the clipped rasters
#     # clipped_rasters = []
#     # for i in range(len(shapefiles)):
#     # #for i in range(10):
#     #     # Clip the raster with the current geometry
#     #     p1 = raster.rio.clip([geometries[i]],shapefiles.crs)
#     #     # Convert the clipped raster to a NumPy array
#     #     p1_np = np.asarray(p1)
#     #     # Append the clipped raster array to the list
#     #     clipped_rasters.append(p1_np)
        
#     return   p1_np_n  

# #%%
# df1=raster_clip(raster_path1, raster_filename1, shapefiles1)
# # df2=raster_clip(raster_path2, raster_filename2, shapefiles2) 
# # df3=raster_clip(raster_path3, raster_filename3, shapefiles3)
# # df4=raster_clip(raster_path4, raster_filename4, shapefiles4)
# # df5=raster_clip(raster_path5, raster_filename5, shapefiles5)
# # df6=raster_clip(raster_path6, raster_filename6, shapefiles6)

# #%%
# df1g=df1[:,:,3]

# #%%
# import sys
# sys.path.append('/S:/mc5545/SA_Drone_data')

# # In[ ]:

# from texture import fastglcm_wrapper

# # Create an instance of the fastglcm_wrapper class
# # Specify the parameters: levels, kernel_size, distance_offset, and angles
# tex2 = fastglcm_wrapper(df1, levels=8, kernel_size=5, distance_offset=5, 
#                         angles=[0, 45, 90, 135])

# correlation = tex2.calculate_glcm_correlation()

# #%%


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




# In[55]:


# p1_np is of shape (4, 553, 552)
#The order is  G, R, RE, NIR for our stack
p1_np
p1_np_n = np.transpose(p1_np, (1, 2, 0))
#p1_np[3]
# Now, p1_np will be of shape (553, 552, 4) with the order of the bands preserved
p1_np_n.shape


# In[ ]:
import sys
sys.path.append('/S:/mc5545/SA_Drone_data')

# In[ ]:

from texture import fastglcm_wrapper

# Create an instance of the fastglcm_wrapper class
# Specify the parameters: levels, kernel_size, distance_offset, and angles
tex2 = fastglcm_wrapper(p1_np_n, levels=8, kernel_size=5, distance_offset=5, 
                        angles=[0, 45, 90, 135])

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




#%%
i=2
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



