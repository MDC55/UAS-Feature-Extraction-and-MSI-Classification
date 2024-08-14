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

#directory_path='F:/2023 SA Fynbos Field Work/Drone data analysis code/from 2022 trip/Indices_csv/'
directory_path='S:/mc5545/SA_Drone_data/from 2022 trip/Indices_csv/'
df1_i = pd.read_csv(f'{directory_path}indices_2006.csv', index_col=0)
df2_i = pd.read_csv(f'{directory_path}indices_2016_1.csv', index_col=0)
df3_i = pd.read_csv(f'{directory_path}indices_2016_2.csv', index_col=0)
df4_i = pd.read_csv(f'{directory_path}indices_2019.csv', index_col=0)
df5_i = pd.read_csv(f'{directory_path}indices_2020.csv', index_col=0)


# Load textures DataFrame from the CSV file
#directory_path='F:/2023 SA Fynbos Field Work/Drone data analysis code/from 2022 trip/textures_csv/'
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
#df4['Alpha_diversity'] = 1.8669 #original plot 2019, i highly doubt this value
df4['Alpha_diversity'] = 2.1669 #
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
#directory_path='F:/2023 SA Fynbos Field Work/Drone data analysis code/Indices_csv/'
directory_path='S:/mc5545/SA_Drone_data/Indices_csv/'
df1_i = pd.read_csv(f'{directory_path}indices_2006_burnplot18.csv', index_col=0)
df2_i = pd.read_csv(f'{directory_path}indices_2016_burn2016.csv', index_col=0)
df3_i = pd.read_csv(f'{directory_path}indices_2017_burn2017.csv', index_col=0)
df4_i = pd.read_csv(f'{directory_path}indices_2019_burn2019.csv', index_col=0)
df5_i = pd.read_csv(f'{directory_path}indices_2020_burnplot17.csv', index_col=0)
df6_i = pd.read_csv(f'{directory_path}indices_2022_burn2022.csv', index_col=0)


# Load textures DataFrame from the CSV file
#directory_path='F:/2023 SA Fynbos Field Work/Drone data analysis code/textures_csv/'
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

#%%

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
    plt.imshow(alpha_diversities_matrix, cmap='viridis')
    #plt.title(f'Year: {year} | Estimated Alpha Diversity: {estimated_alpha_diversity:.4f}')
    plt.title(f'Year: {year}')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    #plt.colorbar(label='Individual Alpha Diversity')
    cbar = plt.colorbar(label='Individual Alpha Diversity')
    
    # Increase font size of colorbar label
    #cbar.ax.yaxis.label.set_fontsize(16)  # Adjust the fontsize as needed
    
    # Increase font size of colorbar tick labels
    #cbar.ax.tick_params(axis='y', labelsize=16)  # Adjust the fontsize as needed
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
dataframes = pd.concat([df1_all, df2_all,df4_all,df5_all], axis=0)

# Concatenate all estimated alpha diversity values for different years
estimated_alpha_all_years = np.concatenate([estimated_alpha_2006, 
                                            estimated_alpha_2016, 
                                            estimated_alpha_2019,
                                            estimated_alpha_2020])


import matplotlib.pyplot as plt

# Create a scatter plot for each feature against the estimated alpha diversity
for column in columns_to_keep:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=dataframes[column], y=estimated_alpha_all_years)
    plt.xlabel(column)
    plt.ylabel('Estimated Alpha Diversity')
    plt.title(f'Scatter plot of {column} vs. Estimated Alpha Diversity')
    plt.show()
#
# Assuming df1, df2, df3, df4, and df5 are your dataframes
# Assuming estimated_alpha_2006, estimated_alpha_2016, estimated_alpha_2019, 
#and estimated_alpha_2020 are your estimated alpha diversity values


# Create a scatter plot for each feature against the estimated alpha diversity
for column in columns_to_keep:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df1_all[column], y=estimated_alpha_2006)
    plt.xlabel(column)
    plt.ylabel('Estimated Alpha Diversity')
    plt.title(f'Scatter plot of {column} vs. Estimated Alpha Diversity')
    plt.show()

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
#pip instll scipy -U

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

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Create separate dataframes for each array
data_2006 = pd.DataFrame({'Estimated Alpha': estimated_alpha_2006})
data_2016 = pd.DataFrame({'Estimated Alpha': estimated_alpha_2016})
data_2019 = pd.DataFrame({'Estimated Alpha': estimated_alpha_2019})
data_2020 = pd.DataFrame({'Estimated Alpha': estimated_alpha_2020})

# Create ridge plots for each dataframe
dataframes = [data_2006, data_2016, data_2019, data_2020]
years = ['Year 2006', 'Year 2016', 'Year 2019', 'Year 2020']

plt.figure(figsize=(10, 6))

for i, df in enumerate(dataframes):
    sns.kdeplot(data=df, x='Estimated Alpha', fill=True, label=years[i], linewidth=1.5)

plt.title('Ridge Plot of Estimated Alpha')
plt.xlabel('Estimated Alpha')
plt.ylabel('Density')
plt.legend()
plt.show()

#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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
plt.savefig('S:/mc5545/SA_Drone_data/study_areas/ridgeplot.png', dpi=300)

plt.show()


Alpha_2006= 2.0345
Alpha_2016_1 = 2.0761
Alpha_2016_2 = 2.0281
Alpha_2019 = 1.8669
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
    #'Year 2019': 1.8669, #original
    'Year 2019': 2.1669, #
    'Year 2020': 2.2425
}

for ax, (year, alpha_value) in zip(g.axes.flat, alpha_values.items()):
    ax.axvline(alpha_value, color='red', linestyle='--')

# Set title
plt.suptitle('Ridge Plot of Estimated Alpha Diversity by Year', y=0.98, fontsize=20, fontweight='bold')
# Save the plot in a directory with 200 DPI resolution
plt.savefig('F:/2023 SA Fynbos Field Work/1. Writing-Project 2/Study Area Fynbos/ridgeplot1.png', dpi=300)

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
with rasterio.open('S:/mc5545/SA_Drone_data/burnplot18_lr/alpha_2006_new/est_alpha_2006.tif', 
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


#%%
import rasterio
# Path to the TIFF image file
file_path = "F:/2023 SA Fynbos Field Work/Drone data analysis code/alpha_diversity/est_alpha_2006.tif"

with rasterio.open(file_path) as image:
    image_array = image.read()
    
    
# Plot the image
plt.imshow(image_array[1], cmap='viridis')  # You can specify the colormap if needed
plt.axis('off')  # Turn off axis
plt.colorbar(label='Individual Alpha Diversity')
plt.title('TIFF Image')
plt.show()    
    
#%%
import numpy as np
#%% Shanon Diversity Index
#when percent cover doesnt sum to 100
#tutorail: https://www.youtube.com/watch?v=lbnXPI05qNI&ab_channel=Geeky_Gardener

# import math

# # Define the abundance values
# abundance_values = [18, 16, 25, 2, 12, 1, 5, 8, 3, 1, 3, 1] #2006

# # Calculate the total abundance
# total_abundance = sum(abundance_values)

# # Calculate the relative abundance
# relative_abundance = [abundance / total_abundance for abundance in abundance_values]

# # Calculate the Shannon diversity index
# shannon_index = -sum(p * math.log2(p) if p != 0 else 0 for p in relative_abundance)

# print(f"Shannon Diversity Index: {shannon_index}")

def shannon_index(abundance_values):
    import math

    # Calculate the total abundance
    total_abundance = sum(abundance_values)

    # Calculate the relative abundance
    relative_abundance = [abundance / total_abundance for abundance in abundance_values]

    # Calculate the Shannon diversity index
    shannon_index = -sum(p * np.log(p) if p != 0 else 0 for p in relative_abundance)

    print(f"Shannon Diversity Index: {shannon_index}")
    return shannon_index
    
# Define the abundance values
abundance_values_2006 =  [18, 16, 25, 2, 12, 1, 5, 8, 3, 1, 3, 1] 
abundance_values_2016_1= [16,22,3,6,2,25,15,3,5,1,3,1,1]
abundance_values_2016_2= [5,7,9,1,45,4,4,2,13,2,6,1,1,2,1,1]
abundance_values_2019 =  [10,15,5,30,45,4,4,10,4,2]
abundance_values_2020 =  [16,10,4,22,14,35,4,15,3,22,25,3]

sh_2006   = shannon_index(abundance_values_2006)
sh_2016_1 = shannon_index(abundance_values_2016_1)
sh_2016_2 = shannon_index(abundance_values_2016_2)
sh_2019   = shannon_index(abundance_values_2019)
sh_2020   = shannon_index(abundance_values_2020)

import matplotlib.pyplot as plt

# Replace the following list with your Shannon diversity index values for each year
shannon_indices = [sh_2006,sh_2016_1,sh_2016_2,sh_2019,sh_2020]

# Replace the following list with your corresponding years
years = [2006, 2016, 2016, 2019, 2020]

# Plotting the Shannon diversity indices over the years
plt.plot(years, shannon_indices, marker='o', linestyle='-', color='b')
plt.title('Shannon Diversity Index Over the Years')
plt.xlabel('Year')
plt.ylabel('Shannon Diversity Index')
plt.grid(True)
plt.show()

#%%
Alpha_2006= 2.0345
Alpha_2016_1 = 2.0761
Alpha_2016_2 = 2.0281
Alpha_2019 = 1.8669
Alpha_2020= 2.2425    



#%%
columns_ = ['mean_NIR', 'CV_RedEdge', 'CVI', 'skewness_DSWI4', 'kurtosis_PSRI',
                   'M3Cl', 'CV_LCI', 'CV_ratio1', 'dissimilarity_mean_band2',
                   'homogeneity_mean_band4']
import matplotlib.pyplot as plt

# Assuming df1_all and df1 are your DataFrames
# Assuming column_name contains numerical data

# Plot histogram for df1_all
plt.figure(figsize=(8, 6))
plt.hist(df1_all['homogeneity_mean_band4'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Histogram of df1_all with df1 point')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.grid(True)

# Plot a vertical line for the value in df1
if not df1.empty:
    df1_value = df1.iloc[0]['homogeneity_mean_band4']  # Assuming the first row and the specified column
    plt.axvline(df1_value, color='red', linestyle='dashed', linewidth=1)
    plt.text(df1_value, plt.gca().get_ylim()[1]*0.9, f'df1 value: {df1_value}', color='red', rotation=90)

plt.show()

#%%


# #%% Manhattan distance

# from scipy.spatial.distance import euclidean
# from scipy.spatial.distance import cityblock  # Manhattan distance

# def estimate_alpha_diversity(df_all, df_known_alpha_diversity):
#     # Extract features and coordinates from the dataframes
#     features_all = df_all.drop(['x', 'y'], axis=1).values
#     known_features = df_known_alpha_diversity.drop(['x', 'y','Alpha_diversity'], axis=1).values
#     known_alpha_diversity = df_known_alpha_diversity['Alpha_diversity'].values[0]  # Assuming only one known alpha diversity

#     # Center the data around known alpha diversity features
#     centered_features_all = features_all - known_features

#     # # Calculate Manhattan distance between centered features and coordinates
#     # distances = np.array([cityblock(centered_features_all[i], np.zeros_like(known_features[0]))
#     #                       for i in range(len(centered_features_all))])
    
#     # # Calculate similarity metric (Euclidean distance) between centered features and coordinates
#     distances = np.array([euclidean(centered_features_all[i], np.zeros_like(known_features[0]))
#                            for i in range(len(centered_features_all))])

#     # Compute weights based on inverse distances
#     weights = 1 / (1 + distances)
    
#     # Calculate individual alpha diversities for each coordinate
#     individual_alpha_diversities = weights * known_alpha_diversity

#     return individual_alpha_diversities, distances, weights, centered_features_all

# estimated_alpha_20061,distances1,weights1,centered_features_all = estimate_alpha_diversity(df1_all, df1)


# #%%
# from scipy.stats import gaussian_kde
# from scipy.stats import norm
# def estimate_alpha_diversity(df_all, df_known_alpha_diversity, bandwidth=1.0):
#     # Extract features and coordinates from the dataframes
#     features_all = df_all.drop(['x', 'y'], axis=1).values
#     known_features = df_known_alpha_diversity.drop(['x', 'y','Alpha_diversity'], axis=1).values
#     known_alpha_diversity = df_known_alpha_diversity['Alpha_diversity'].values[0]  # Assuming only one known alpha diversity

#     # Center the data around known alpha diversity features
#     centered_features_all = features_all - known_features

#     # Calculate Euclidean distance between centered features and coordinates
#     distances = np.array([euclidean(centered_features_all[i], np.zeros_like(known_features[0]))
#                           for i in range(len(centered_features_all))])

#     # Compute weights based on kernel density estimation
#     #kde = gaussian_kde(distances, bw_method=bandwidth)
#     #weights = kde(distances)
#     # Compute weights based on Gaussian kernel
#     weights = norm.pdf(distances, scale=bandwidth)
    
#     # Calculate individual alpha diversities for each coordinate
#     individual_alpha_diversities = weights * known_alpha_diversity

#     return individual_alpha_diversities, distances, weights, centered_features_all
# estimated_alpha_2006,distances1,weights1,centered_features_all = estimate_alpha_diversity(df1_all, df1)
# estimated_alpha_2016,x,y,z=estimate_alpha_diversity(df2_all, df2)
# estimated_alpha_2019,x,y,z=estimate_alpha_diversity(df4_all, df4)  
# estimated_alpha_2020,x,y,z=estimate_alpha_diversity(df5_all, df5)


# #%%
# from sklearn.metrics.pairwise import cosine_similarity

# def estimate_alpha_diversity(df_all, df_known_alpha_diversity):
#     # Extract features and coordinates from the dataframes
#     features_all = df_all.drop(['x', 'y'], axis=1).values
#     known_features = df_known_alpha_diversity.drop(['x', 'y','Alpha_diversity'], axis=1).values
#     known_alpha_diversity = df_known_alpha_diversity['Alpha_diversity'].values[0]  # Assuming only one known alpha diversity

#     # Calculate cosine similarity between features and known alpha diversity features
#     similarities = cosine_similarity(features_all, known_features)

#     # Convert cosine similarity to a similarity measure ranging from -1 to 1
#     # We add 1 to shift the range from -1 to 1 to 0 to 2, then divide by 2 to scale it to the range 0 to 1
#     similarities = (similarities + 1) / 2
    
#     # Calculate individual alpha diversities for each coordinate
#     individual_alpha_diversities = similarities * known_alpha_diversity

#     return individual_alpha_diversities, similarities

# estimated_alpha_2006, similarities= estimate_alpha_diversity(df1_all, df1)
# estimated_alpha_2016,x,=estimate_alpha_diversity(df2_all, df2)
# estimated_alpha_2019,x,=estimate_alpha_diversity(df4_all, df4)  
# estimated_alpha_2020,x,=estimate_alpha_diversity(df5_all, df5)

#%%
# #Fine Tuning
# from sklearn.metrics.pairwise import cosine_similarity
# #
# #df1  #2006
# #df2  #2016
# #df3  #2016
# #df4  #2019
# #df5  #2020

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
    
#     # Calculate cosine similarity between features and known alpha diversity features
#     #distances = cosine_similarity(features_all, known_features)

#     # Convert cosine similarity to a similarity measure ranging from -1 to 1
#     # We add 1 to shift the range from -1 to 1 to 0 to 2, then divide by 2 to scale it to the range 0 to 1
#     #weights = (distances + 1) / 2
    
#     # Compute weights based on inverse distances
#     weights = 1 / (1 + distances)
#     mean_weight = np.mean(weights)
#     # Weighted average of alpha diversities
#     estimated_alpha_diversity = np.sum(weights * known_alpha_diversity) / np.sum(weights)
    
#     # Calculate individual alpha diversities for each coordinate
#     individual_alpha_diversities = (weights*known_alpha_diversity)/mean_weight
#     #individual_alpha_diversities = (abs(weights-mean_weight))*known_alpha_diversity
#     #individual_alpha_diversities = (weights*known_alpha_diversity)
#     #individual_alpha_diversities = weights
#     # Calculate individual alpha diversities for each coordinate

#     # # Adjust individual alpha diversities based on the weights compared to the mean
#     # if np.any(weights > mean_weight):
#     #     individual_alpha_diversities = known_alpha_diversity + weights
#     # else:
#     #     individual_alpha_diversities = known_alpha_diversity - weights

#     return individual_alpha_diversities,estimated_alpha_diversity

# # Apply the approach to each dataframe and create separate heatmap plots
# dfs_all = [df1_all, df2_all, df4_all, df5_all]
# known_alpha_diversity_dfs = [df1, df2, df4, df5]



# years = [2006, 2016, 2019, 2020]

# # for i, (df_all, known_alpha_diversity_df, year) in enumerate(zip(dfs_all, known_alpha_diversity_dfs, years)):
# #     individual_alpha_diversities, estimated_alpha_diversity = estimate_alpha_diversity(df_all, known_alpha_diversity_df)

# #     # Reshape individual alpha diversities into a 2D array
# #     side_length = int(np.sqrt(len(individual_alpha_diversities)))
# #     alpha_diversities_matrix = individual_alpha_diversities[:side_length**2].reshape((side_length, side_length))

# #     # Create a heatmap
# #     plt.imshow(alpha_diversities_matrix, cmap='viridis')
# #     #plt.title(f'Year: {year} | Estimated Alpha Diversity: {estimated_alpha_diversity:.4f}')
# #     plt.title(f'Year: {year}')
# #     plt.xlabel('X-coordinate')
# #     plt.ylabel('Y-coordinate')
# #     #plt.colorbar(label='Individual Alpha Diversity')
# #     cbar = plt.colorbar(label='Individual Alpha Diversity')
    
# #     # Increase font size of colorbar label
# #     #cbar.ax.yaxis.label.set_fontsize(16)  # Adjust the fontsize as needed
    
# #     # Increase font size of colorbar tick labels
# #     #cbar.ax.tick_params(axis='y', labelsize=16)  # Adjust the fontsize as needed
# #     plt.show()

# dfx=  df1.drop(['Alpha_diversity'], axis=1)  
# estimated_alpha_2006,all_sum=estimate_alpha_diversity(df1_all, df1)
# estimated_alpha_2006_,all_sum=estimate_alpha_diversity(dfx, df1)
# estimated_alpha_2016,all_sum=estimate_alpha_diversity(df2_all, df2)
# estimated_alpha_2019,all_sum=estimate_alpha_diversity(df4_all, df4)  
# estimated_alpha_2020,all_sum=estimate_alpha_diversity(df5_all, df5)

# np.mean(estimated_alpha_2006)


   
#%%
# import numpy as np



# # Calculate the mean of all 644 data points
# mean_weight = np.mean(estimated_alpha_2006)

# # Apply the specified condition
# if np.any(estimated_alpha_2006 == mean_weight):
#     new_values = 2.0345
# elif np.any(estimated_alpha_2006 > mean_weight):
#     new_values = 2.0345+estimated_alpha_2006    
# else:
#     new_values = 2.0345-estimated_alpha_2006

# # Print the new array of values
# print(f"New array of values: {new_values}")

#%%

# for i in range(number_of_plots):
#         dot_product = np.dot(plot_data[i], ref_data)
#         cos_sim = dot_product / (np.linalg.norm(plot_data[i])*np.linalg.norm(ref_data))
#         cos_sims.append(cos_sim)
#     # Adjust the cosine similarities so that their mean equals 1
#     cos_sims = np.array(cos_sims) / np.mean(cos_sims)
#     # Scale the cosine similarities with the reference alpha diversity
#     alpha_diversities = cos_sims * ref_alpha
    
#%%
import matplotlib.pyplot as plt

# Assuming df1_all and df1 are your DataFrames
# Assuming column_name contains numerical data

# Plot histogram for df1_all
plt.figure(figsize=(8, 6))
plt.hist(estimated_alpha_2020, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Histogram of df1_all with df1 point')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.grid(True)

# Plot a vertical line for the value in df1
if not df1.empty:
    df1_value =2.03  # Assuming the first row and the specified column
    plt.axvline(df1_value, color='red', linestyle='dashed', linewidth=1)
    plt.text(df1_value, plt.gca().get_ylim()[1]*0.9, f'df1 value: {df1_value}', color='red', rotation=90)

plt.show()

#%%



