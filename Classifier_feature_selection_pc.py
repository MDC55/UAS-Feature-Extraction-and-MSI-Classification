#!/usr/bin/env python
# coding: utf-8

# In[27]:


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



# Load indices DataFrame from the CSV file
directory_path='F:/2023 SA Fynbos Field Work/Drone data analysis code/Indices_csv/'
df1_i = pd.read_csv(f'{directory_path}indices_2006_burnplot18.csv', index_col=0)
df2_i = pd.read_csv(f'{directory_path}indices_2016_burn2016.csv', index_col=0)
df3_i = pd.read_csv(f'{directory_path}indices_2017_burn2017.csv', index_col=0)
df4_i = pd.read_csv(f'{directory_path}indices_2019_burn2019.csv', index_col=0)
df5_i = pd.read_csv(f'{directory_path}indices_2020_burnplot17.csv', index_col=0)
df6_i = pd.read_csv(f'{directory_path}indices_2022_burn2022.csv', index_col=0)


# Load textures DataFrame from the CSV file
directory_path='F:/2023 SA Fynbos Field Work/Drone data analysis code/textures_csv/'
df1_t = pd.read_csv(f'{directory_path}textures_2006_burnplot18.csv', index_col=0)
df2_t = pd.read_csv(f'{directory_path}textures_2016_burn2016.csv', index_col=0)
df3_t = pd.read_csv(f'{directory_path}textures_2017_burn2017.csv', index_col=0)
df4_t = pd.read_csv(f'{directory_path}textures_2019_burn2019.csv', index_col=0)
df5_t = pd.read_csv(f'{directory_path}textures_2020_burnplot17.csv', index_col=0)
df6_t = pd.read_csv(f'{directory_path}textures_2022_burn2022.csv', index_col=0)

# Concatenate the indices and textures features into a new df along the columns (axis=1) 
df1 = pd.concat([df1_i, df1_t], axis=1)
df2 = pd.concat([df2_i, df2_t], axis=1)
df3 = pd.concat([df3_i, df3_t], axis=1)
df4 = pd.concat([df4_i, df4_t], axis=1)
df5 = pd.concat([df5_i, df5_t], axis=1)
df6 = pd.concat([df6_i, df6_t], axis=1)

dataframes = [df1, df2, df3, df4, df5, df6]

df1['Year'] = 2006
df2['Year'] = 2016
df3['Year'] = 2017
df4['Year'] = 2019
df5['Year'] = 2020
df6['Year'] = 2022

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Concatenate DataFrames
concatenated_df = pd.concat(dataframes)
# List of columns to drop
columns_to_drop = ['x', 'y']


# Drop the specified columns
X= concatenated_df.drop(columns=columns_to_drop)


# # Assuming df is your DataFrame
# Selectednames=['CV_Green', 'CV_ReCI', 'BAI', 'CVI', 'sCCCI', 'kurtosis_sCCCI',
#        'CV_ratio1', 'homogeneity_CV_band3', 'homogeneity_skewness_band3',
#        'correlation_mean_band3','mean_Green','mean_Red','mean_RedEdge','Year']



# Keep only the selected columns
#X = concatenated_df.loc[:, Selectednames]


# X= concatenated_df.drop(columns="x")
# X= concatenated_df.drop(columns="y")
# Extract features (X) and target variable (y)
X = X.drop("Year", axis=1)  # Assuming "year" is the column containing the target variable

y = concatenated_df["Year"]

del (df1,df2,df3,df4,df5,df6,df1_i,df2_i,df3_i,df4_i,df5_i,df6_i,
     df1_t,df2_t,df3_t,df4_t,df5_t,df6_t)

#%%
import pandas as pd




# In[10]:


from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Assuming you have your data (X, y) loaded

# Standardize the features (important for some algorithms)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Visualize the data in 2D using a scatter plot
plt.figure(figsize=(8, 6))
#https://matplotlib.org/stable/users/explain/colors/colormaps.html
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='Paired', edgecolors='k', s=50) #viridis
plt.title('t-SNE Visualization',fontsize=18)
plt.xlabel('t-SNE Component 1',fontsize=16)
plt.ylabel('t-SNE Component 2',fontsize=16)
plt.xticks(fontsize=16)  # Increase font size of x-axis tick labels
plt.yticks(fontsize=16)  # Increase font size of y-axis tick labels
plt.legend(*scatter.legend_elements(), title='Class',fontsize=14)

# Save the plot in a directory with 200 DPI resolution
plt.savefig('F:/2023 SA Fynbos Field Work/Study Area Fynbos/Burn Years.png', dpi=200)

plt.show()


#%%

from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import numpy as np


# Apply SMOTE to the training set
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Standardize the features (important for some algorithms)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

# Apply t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Visualize the data in 2D using a scatter plot
plt.figure(figsize=(8, 6))
#https://matplotlib.org/stable/users/explain/colors/colormaps.html
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_resampled, cmap='Paired', edgecolors='k', s=50) #viridis
plt.title('t-SNE Visualization')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend(*scatter.legend_elements(), title='Class')
# Save the plot in a directory with 200 DPI resolution
#plt.savefig('F:/2023 SA Fynbos Field Work/Study Area Fynbos/Burn Years.png', dpi=200)

plt.show()

# In[7]:


from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to the training set
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Apply standard scaling to the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Create SVM classifier
svm_classifier = SVC(kernel='linear', random_state=42)

# Train the classifier
svm_classifier.fit(X_train_scaled, y_train_resampled)

# Make predictions on the scaled test set
y_pred_svm = svm_classifier.predict(X_test_scaled)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred_svm)
classification_report_result = classification_report(y_test, y_pred_svm)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report_result)


#%%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from mlxtend.feature_selection import SequentialFeatureSelector
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import numpy as np

# Assuming you have your data (X, y) loaded

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply SMOTE to the training set
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Apply standard scaling to the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)


# Create an SVM classifier
svm_classifier = SVC(kernel='rbf')  # You can specify the kernel you want to use

# Apply sequential forward feature selection from mlxtend to the SVM classifier
num_features_to_select = 10  # Specify the number of features you want to select
sfs_selector = SequentialFeatureSelector(svm_classifier, 
                                         k_features=num_features_to_select, 
                                         forward=True, floating=False, verbose=2, 
                                         scoring='accuracy', cv=5)
X_train_selected = sfs_selector.fit_transform(X_train_scaled, y_train_resampled)
X_test_selected = sfs_selector.transform(X_test_scaled)

# Train SVM classifier on the selected features
svm_classifier.fit(X_train_selected, y_train_resampled)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test_selected)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy on the test set: {accuracy:.2f}')

# Get the selected feature indices
selected_feature_indices = np.array(sfs_selector.k_feature_idx_)
print('Selected feature indices:', selected_feature_indices)

# Get the names of the selected features
selected_feature_names = X.columns[selected_feature_indices]  # Assuming X is a DataFrame
print('Selected feature names:', selected_feature_names)

# Plot the results on tranning set
fig1 = plot_sfs(sfs_selector.get_metric_dict(), kind='std_dev')
plt.ylim([0.5, 1])

plt.title('Sequential Forward Selection (SVM)', fontsize=14, fontweight='bold')
plt.xlabel('Number of Features', fontsize=16, fontweight='bold')
plt.ylabel('Performance Metric', fontsize=16, fontweight='bold')
plt.grid()

# Make x and y ticks bold
plt.gca().tick_params(axis='x', which='both', labelsize=14, width=2)
plt.gca().tick_params(axis='y', which='both', labelsize=14, width=2)

plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')

plt.show()



#%%

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from mlxtend.feature_selection import SequentialFeatureSelector
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import numpy as np

# Assuming you have your data (X, y) loaded

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features (important for some algorithms)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create an SVM classifier
svm_classifier = SVC(kernel='rbf')  # You can specify the kernel you want to use

# Apply sequential forward feature selection from mlxtend to the SVM classifier
num_features_to_select = 7  # Specify the number of features you want to select
sfs_selector = SequentialFeatureSelector(svm_classifier, 
                                         k_features=num_features_to_select,
                                         #k_features="best", 
                                         forward=True, floating=False, verbose=2, 
                                         scoring='accuracy', cv=5)
X_train_selected = sfs_selector.fit_transform(X_train_scaled, y_train)
X_test_selected = sfs_selector.transform(X_test_scaled)

# Train SVM classifier on the selected features
svm_classifier.fit(X_train_selected, y_train)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test_selected)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy on the test set: {accuracy:.2f}')

# Get the selected feature indices
selected_feature_indices = np.array(sfs_selector.k_feature_idx_)
print('Selected feature indices:', selected_feature_indices)

# Get the names of the selected features
selected_feature_names = X.columns[selected_feature_indices]  # Assuming X is a DataFrame
print('Selected feature names:', selected_feature_names)

# Plot the results on tranning set
fig1 = plot_sfs(sfs_selector.get_metric_dict(), kind='std_dev')
plt.ylim([0.5, 1])

plt.title('Sequential Forward Selection (SVM)', fontsize=14, fontweight='bold')
plt.xlabel('Number of Features', fontsize=16, fontweight='bold')
plt.ylabel('Performance Metric', fontsize=16, fontweight='bold')
plt.grid()

# Make x and y ticks bold
plt.gca().tick_params(axis='x', which='both', labelsize=14, width=2)
plt.gca().tick_params(axis='y', which='both', labelsize=14, width=2)

plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')

plt.show()

#%%
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import precision_recall_fscore_support

# Assuming y_test and y_pred are your actual and predicted labels respectively
kappa = cohen_kappa_score(y_test, y_pred)
print("Cohen's Kappa Coefficient:", "{:.2f}%".format(kappa * 100))
# Assuming y_test and y_pred are your actual and predicted labels respectively
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

print ("Precision:" "{:.2f}%".format(precision * 100))
print("Recall:", "{:.2f}%".format(recall * 100))
print("F1 Score:", "{:.2f}%".format(f1_score * 100))

# Print classification report
print("Classification Report SVM:")
print(classification_report(y_test, y_pred))

#%% Inspecting the results

sfs_selector.subsets_
metric_dict=sfs_selector.get_metric_dict(confidence_interval=0.95)
metric_dict
df_results=pd.DataFrame.from_dict(metric_dict).T


# Assuming df_results is your DataFrame and X is your feature DataFrame

for i in range(len(df_results.index)):
    z = np.array(df_results.feature_idx[i+1])
    print('Selected feature indices:', z)
    # Get the names of the selected features
    selected_feature_names = X.columns[z]  # Assuming X is a DataFrame
    print('Selected feature names:', selected_feature_names)
    # Replace the feature_names column in df_results
    df_results.at[i+1, 'feature_names'] = selected_feature_names.tolist()

print(df_results)

# Save df_results to a CSV file in a directory
directory_path = 'F:/2023 SA Fynbos Field Work/Classifier results/'
csv_file_path = directory_path + 'RF.csv'
df_results.to_csv(csv_file_path, index=False)
print("DataFrame saved as CSV to:", csv_file_path)
#%%


#%% RF
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from mlxtend.feature_selection import SequentialFeatureSelector
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import numpy as np

# Assuming you have your data (X, y) loaded

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features (important for some algorithms)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a Decision Tree classifier
random_forest_classifier = RandomForestClassifier(random_state=42)

# Apply sequential forward feature selection from mlxtend to the Random Forest classifier
num_features_to_select = 10  # Specify the number of features you want to select
sfs_selector = SequentialFeatureSelector(random_forest_classifier, 
                                         k_features=num_features_to_select, 
                                         forward=True, floating=False, verbose=2, 
                                         scoring='accuracy', cv=5)
X_train_selected = sfs_selector.fit_transform(X_train_scaled, y_train)
X_test_selected = sfs_selector.transform(X_test_scaled)

# Train Random Forest classifier on the selected features
random_forest_classifier.fit(X_train_selected, y_train)

# Make predictions on the test set
y_pred = random_forest_classifier.predict(X_test_selected)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy on the test set: {accuracy:.2f}')

# Get the selected feature indices
selected_feature_indices = np.array(sfs_selector.k_feature_idx_)
print('Selected feature indices:', selected_feature_indices)

# Get the names of the selected features
selected_feature_names = X.columns[selected_feature_indices]  # Assuming X is a DataFrame
print('Selected feature names:', selected_feature_names)


# Plot the results on tranning set
fig1 = plot_sfs(sfs_selector.get_metric_dict(), kind='std_dev')
plt.ylim([0.5, 1])

plt.title('Sequential Forward Selection (Random Forest)', fontsize=14, fontweight='bold')
plt.xlabel('Number of Features', fontsize=16, fontweight='bold')
plt.ylabel('Performance Metric', fontsize=16, fontweight='bold')
plt.grid()

# Make x and y ticks bold
plt.gca().tick_params(axis='x', which='both', labelsize=14, width=2)
plt.gca().tick_params(axis='y', which='both', labelsize=14, width=2)

plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')

plt.show()
#%%

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Assuming you have your data (X, y) loaded

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features (important for some algorithms)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a GradientBoosting classifier
gradient_boosting_classifier = GradientBoostingClassifier(random_state=42)

# Train the classifier
gradient_boosting_classifier.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = gradient_boosting_classifier.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#%%
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score
# from mlxtend.feature_selection import SequentialFeatureSelector
# import numpy as np
# from scipy.stats import ttest_rel

# # Initialize lists to store accuracy scores
# accuracy_scores_fs = []
# accuracy_scores_no_fs = []

# # Number of iterations
# num_iterations = 100

# for _ in range(num_iterations):
#     # Split the data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=_)

#     # Standardize the features (important for some algorithms)
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     # Create an SVM classifier
#     svm_classifier = SVC(kernel='rbf')

#     # Apply sequential forward feature selection from mlxtend to the SVM classifier
#     num_features_to_select = 10
#     sfs_selector = SequentialFeatureSelector(svm_classifier, 
#                                              k_features=num_features_to_select, 
#                                              forward=True, floating=False, verbose=0, 
#                                              scoring='accuracy', cv=5)
#     X_train_selected = sfs_selector.fit_transform(X_train_scaled, y_train)
#     X_test_selected = sfs_selector.transform(X_test_scaled)

#     # Train SVM classifier on the selected features
#     svm_classifier.fit(X_train_selected, y_train)

#     # Make predictions on the test set with feature selection
#     y_pred_fs = svm_classifier.predict(X_test_selected)

#     # Evaluate performance with feature selection and store accuracy score
#     accuracy_fs = accuracy_score(y_test, y_pred_fs)
#     accuracy_scores_fs.append(accuracy_fs)

#     # Make predictions on the test set without feature selection
#     y_pred_no_fs = svm_classifier.predict(X_test_scaled)

#     # Evaluate performance without feature selection and store accuracy score
#     accuracy_no_fs = accuracy_score(y_test, y_pred_no_fs)
#     accuracy_scores_no_fs.append(accuracy_no_fs)

# # Perform a paired t-test to compare performances
# t_statistic, p_value = ttest_rel(accuracy_scores_fs, accuracy_scores_no_fs)

# print(f'T-statistic: {t_statistic:.2f}')
# print(f'P-value: {p_value:.4f}')
