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

#510 Features in Total 
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
df1 = pd.concat([df1_i, df1_t], axis=1)
df2 = pd.concat([df2_i, df2_t], axis=1)
df3 = pd.concat([df3_i, df3_t], axis=1)
df4 = pd.concat([df4_i, df4_t], axis=1)
df5 = pd.concat([df5_i, df5_t], axis=1)
df6 = pd.concat([df6_i, df6_t], axis=1)

dataframes = [df1, df2, df3, df4, df5, df6]

# df1['Year'] = 2006
# df2['Year'] = 2016
# df3['Year'] = 2017
# df4['Year'] = 2019
# df5['Year'] = 2020
# df6['Year'] = 2022

df1['Year'] = 1
df2['Year'] = 2
df3['Year'] = 3
df4['Year'] = 4
df5['Year'] = 5
df6['Year'] = 6


# Concatenate DataFrames
concatenated_df = pd.concat(dataframes)
# List of columns to drop
columns_to_drop = ['x', 'y']

# Drop the specified columns
X= concatenated_df.drop(columns=columns_to_drop)
# X= concatenated_df.drop(columns="x")
# X= concatenated_df.drop(columns="y")
# Extract features (X) and target variable (y)
X = X.drop("Year", axis=1)  # Assuming "year" is the column containing the target variable

y = concatenated_df["Year"]

del (df1,df2,df3,df4,df5,df6,df1_i,df2_i,df3_i,df4_i,df5_i,df6_i,
     df1_t,df2_t,df3_t,df4_t,df5_t,df6_t)

#%%
#%% SMOTE ;SMOTE: Synthetic Minority Oversampling Technique
#https://www.analyticsvidhya.com/blog/2020/10/overcoming-class-imbalance-using-smote-techniques/
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from collections import Counter


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to the training set
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Visualize the distribution before and after SMOTE
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Before SMOTE
ax1.bar(Counter(y_train).keys(), Counter(y_train).values(), color='blue')
ax1.set_title('Distribution Before SMOTE')
ax1.set_xlabel('Year Category')
ax1.set_ylabel('Count')

# After SMOTE
ax2.bar(Counter(y_train_resampled).keys(), Counter(y_train_resampled).values(), color='green')
ax2.set_title('Distribution After SMOTE')
ax2.set_xlabel('Year Category')
ax2.set_ylabel('Count')

plt.show()

#%% 3 Class in one graph

# Before SMOTE
class_1_before_smote = X_train[y_train == 1]
class_2_before_smote = X_train[y_train == 2]
class_3_before_smote = X_train[y_train == 3]
# After SMOTE
class_1_after_smote = X_train_resampled[y_train_resampled == 1]
class_2_after_smote = X_train_resampled[y_train_resampled == 2]
class_3_after_smote = X_train_resampled[y_train_resampled == 3]
# Choose two features for the scatter plot
feature1 = 'mean_Green'  # Replace with your actual feature names
feature2 = 'mean_RedEdge'

# Scatter plot before SMOTE
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(class_1_before_smote[feature1], class_1_before_smote[feature2], label='Class 1', alpha=0.5)
plt.scatter(class_2_before_smote[feature1], class_2_before_smote[feature2], label='Class 2', alpha=0.5)
plt.scatter(class_3_before_smote[feature1], class_3_before_smote[feature2], label='Class 3', alpha=0.5)
plt.title('Scatter Plot Before SMOTE')
plt.xlabel(feature1)
plt.ylabel(feature2)
plt.legend()

# Scatter plot after SMOTE
plt.subplot(1, 2, 2)
plt.scatter(class_1_after_smote[feature1], class_1_after_smote[feature2], label='Class 1', alpha=0.5)
plt.scatter(class_2_after_smote[feature1], class_2_after_smote[feature2], label='Class 2', alpha=0.5)
plt.scatter(class_3_after_smote[feature1], class_3_after_smote[feature2], label='Class 3', alpha=0.5)
plt.title('Scatter Plot After SMOTE')
plt.xlabel(feature1)
plt.ylabel(feature2)
plt.legend()

plt.tight_layout()
plt.show()

#%% trying to see in 3D, not satisfied

#%% ALL class in one graph, not satisfied


#%% All seperate class in one graph
# Define the number of classes
num_classes = 6

# Choose two features for the scatter plot
feature1 = 'mean_Green'  # Replace with your actual feature names
feature2 = 'mean_RedEdge'

# Before SMOTE
before_smote_data = [X_train[y_train == class_num] for class_num in range(1, num_classes + 1)]

# After SMOTE
after_smote_data = [X_train_resampled[y_train_resampled == class_num] for class_num in range(1, num_classes + 1)]

# Plotting
fig, axes = plt.subplots(2, num_classes, figsize=(18, 8))

color_before = 'blue'  # Change to your desired color for before SMOTE
color_after = 'orange'  # Change to your desired color for after SMOTE

for class_num, (before_data, after_data) in enumerate(zip(before_smote_data, after_smote_data), start=1):
    # Scatter plot before SMOTE
    axes[0, class_num - 1].scatter(before_data[feature1], before_data[feature2], label=f'Class {class_num}', alpha=0.5, color=color_before)
    axes[0, class_num - 1].set_title(f'Scatter Plot Before SMOTE - Class {class_num}')
    axes[0, class_num - 1].set_xlabel(feature1)
    axes[0, class_num - 1].set_ylabel(feature2)
    axes[0, class_num - 1].legend()

    # Scatter plot after SMOTE
    axes[1, class_num - 1].scatter(after_data[feature1], after_data[feature2], label=f'Class {class_num}', alpha=0.5, color=color_after)
    axes[1, class_num - 1].set_title(f'Scatter Plot After SMOTE - Class {class_num}')
    axes[1, class_num - 1].set_xlabel(feature1)
    axes[1, class_num - 1].set_ylabel(feature2)
    axes[1, class_num - 1].legend()

plt.tight_layout()
plt.show()

#%% SVM in SMOTE
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to the training set
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


# Split the resampled data into training and testing sets
#X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(X_train_resampled, y_train_resampled, test_size=0.2, random_state=42)

# Create SVM classifier
svm_classifier = SVC(kernel='linear', random_state=42)

# Train the classifier
svm_classifier.fit(X_train_resampled, y_train_resampled)

# Make predictions on the test set
y_pred_svm = svm_classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred_svm)
classification_report_result = classification_report(y_test, y_pred_svm)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report_result)



#%% Boruta
#https://medium.com/analytics-vidhya/is-this-the-best-feature-selection-algorithm-borutashap-8bc238aa1677
#%%

#%%

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features (important for SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#%%
#https://github.com/codingnest/FeatureSelection/blob/master/Data%20Science%20Lifecycle%20-%20Feature%20Selection%20(Filter%2C%20Wrapper%2C%20Embedded%20and%20Hybrid%20Methods).ipynb
import seaborn as sns
corr_mat = X_train.corr()
fig, ax = plt.subplots()
fig.set_size_inches(12,12)
sns.heatmap(corr_mat, cmap='magma');

#%% Filter mmethod 2/19/2024

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix



# Assuming you have X (features) and y (target variable)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Specify the number of features you want to select
k = 100

# Initialize SelectKBest with the f_classif scoring function
selector = SelectKBest(f_classif, k=k)

# Fit the selector to your training data and transform both training and testing sets
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

# Initialize SVM classifier
svm_classifier = SVC()

# Train the SVM classifier on the selected features
svm_classifier.fit(X_train_selected, y_train)

# Make predictions on the test set using the model trained on selected features
y_pred = svm_classifier.predict(X_test_selected)

# Evaluate the performance of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on the test set: {accuracy}")


years = [2006, 2016, 2017, 2019, 2020, 2022]

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Normalize the confusion matrix
conf_matrix_normalized = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)

# Plot normalized confusion matrix
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)  # Adjust font size for better readability
sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2', cmap='Blues', cbar=True, square=True,
            xticklabels=years, yticklabels=years, annot_kws={"size": 12})

plt.title('Normalized Confusion Matrix')
plt.xlabel('Predicted Year')
plt.ylabel('True Year')
plt.show()




#%% Using Cross-Validation: Filter mmethod 

from sklearn.model_selection import cross_val_score

# Define a range of k values to try
k_values = [5, 10, 15, 20, 30, 50, 100]

# Initialize lists to store cross-validation scores
cv_scores = []

# Perform cross-validation for each k value
for k in k_values:
    selector = SelectKBest(f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    svm_classifier = SVC()
    scores = cross_val_score(svm_classifier, X_train_selected, y_train, cv=10)  # Adjust the number of folds as needed
    cv_scores.append(scores.mean())

# Find the k value with the highest cross-validation score
best_k = k_values[cv_scores.index(max(cv_scores))]

print(f"Optimal k value: {best_k}")

# Train the final model with the selected k value
final_selector = SelectKBest(f_classif, k=best_k)
X_train_final_selected = final_selector.fit_transform(X_train_scaled, y_train)
svm_classifier_final = SVC()
svm_classifier_final.fit(X_train_final_selected, y_train)

# Evaluate the performance on the test set
X_test_final_selected = final_selector.transform(X_test_scaled)
y_pred_final = svm_classifier_final.predict(X_test_final_selected)
accuracy_final = accuracy_score(y_test, y_pred_final)
print(f"Accuracy on the test set with optimal k value: {accuracy_final}")


#%% Grid search CV:Filter mmethod 
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

# Define a range of k values to search over
k_values = [5, 10, 15, 20, 30, 50, 100]

# Create a pipeline with SelectKBest and SVM classifier
pipeline = Pipeline([
    ('selector', SelectKBest(f_classif)),
    ('classifier', SVC())
])

# Set up the parameter grid for GridSearchCV
param_grid = {
    'selector__k': k_values,
    'classifier__C': [0.1, 1, 10],  # You can adjust other SVM hyperparameters
    'classifier__gamma': ['scale', 'auto'],
    'classifier__kernel': ['linear', 'poly', 'rbf', 'sigmoid']  # Include different kernel options
}

# Initialize GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5)  # Adjust the number of folds as needed

# Fit the grid search to the training data
grid_search.fit(X_train_scaled, y_train)

# Print the best parameters and corresponding accuracy
print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)

# Evaluate the performance on the test set using the best model from grid search
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test_scaled)
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"Accuracy on the test set with the best model: {accuracy_best}")

years = [2006, 2016, 2017, 2019, 2020, 2022]

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Normalize the confusion matrix
conf_matrix_normalized = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)

# Plot normalized confusion matrix
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)  # Adjust font size for better readability
sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2', cmap='Blues', cbar=True, square=True,
            xticklabels=years, yticklabels=years, annot_kws={"size": 12})

plt.title('Normalized Confusion Matrix')
plt.xlabel('Predicted Year')
plt.ylabel('True Year')
plt.show()

#%% PCA visualization

print("Shape of X_train_scaled:", X_train_scaled.shape)
print("Features used for plotting:", X_test_scaled.shape[1])

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Assuming you have X_train_scaled, X_test_scaled, y_train, and y_test
# ... (your previous code)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Fit SVC to the training data with the best parameters
best_params = {'C': 0.1, 'gamma': 'scale', 'kernel': 'linear'}
svm_classifier = SVC(**best_params)
svm_classifier.fit(X_train_pca, y_train)

# Evaluate the performance on the test set
y_pred = svm_classifier.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on the test set: {accuracy}")

# Plot decision boundary
h = 0.02  # step size in the mesh
x_min, x_max = X_test_pca[:, 0].min() - 1, X_test_pca[:, 0].max() + 1
y_min, y_max = X_test_pca[:, 1].min() - 1, X_test_pca[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = svm_classifier.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues, alpha=0.8)

# Scatter plot of data points
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, edgecolors='k', cmap=plt.cm.Paired)

# Set plot labels and title
plt.title('Decision Boundary of SVM Classifier (PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.show()


#%% Learning curve:Filter mmethod 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve

# Assuming you have X_train_scaled, y_train, and your optimal model (best_model) from the grid search

# Create a learning curve function
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# Plot learning curves
plot_learning_curve(best_model, "Learning Curves", X_train_scaled, y_train, cv=10, n_jobs=-1)

plt.show()
#%%



#%% Plotting Performance vs. Number of Features:

import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Assuming you have X_train_scaled, X_test_scaled, y_train, and y_test
k_values = [5, 10, 15, 20, 30, 50, 100]

accuracies = []

for k in k_values:
    selector = SelectKBest(f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)

    svm_classifier = SVC()
    svm_classifier.fit(X_train_selected, y_train)

    y_pred = svm_classifier.predict(X_test_selected)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

plt.plot(k_values, accuracies, marker='o')
plt.xlabel('Number of Features (k)')
plt.ylabel('Accuracy')
plt.title('Model Performance vs. Number of Features')
plt.show()
#%%
#https://www.visual-design.net/post/feature-selection-and-eda-in-machine-learning

#%%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Assuming you have X_train_scaled, X_test_scaled, y_train, and y_test
k_values = [5, 10, 15, 20, 30, 50, 100]

# Choose a specific value of k for which you want to visualize selected features
selected_k = 20

# Initialize SelectKBest with f_classif scoring and k features
selector = SelectKBest(f_classif, k=selected_k)

# Fit and transform the training data using the selected features
X_train_selected = selector.fit_transform(X_train_scaled, y_train)

# Get the indices of the selected features
selected_feature_indices = selector.get_support(indices=True)

# Plot a bar chart to visualize selected features
plt.bar(range(len(selected_feature_indices)), selector.scores_[selected_feature_indices], tick_label=selected_feature_indices)
plt.xlabel('Feature Index')
plt.ylabel('ANOVA F-statistic')
plt.title(f'Selected Features for k={selected_k}')
plt.show()

#%%
#%%
#
'''
Wrapper methods for feature selection involve using a specific machine learning model 
(such as SVM in your case) as part of the feature selection process. Common wrapper
 methods include Recursive Feature Elimination (RFE) and Forward/Backward Stepwise
 Selection. Here, I'll provide an example using RFE with an SVM classifier:'''

from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Assuming you have your data (X, y) loaded

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features (important for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create an SVM classifier
svm_classifier = SVC(kernel='linear')  # You can specify the kernel you want to use

# Use RFE for feature selection
num_features_to_select = 10  # Specify the number of features you want to select
rfe_selector = RFE(svm_classifier, n_features_to_select=num_features_to_select)
X_train_selected = rfe_selector.fit_transform(X_train_scaled, y_train)
X_test_selected = rfe_selector.transform(X_test_scaled)

# Train SVM classifier on the selected features
svm_classifier.fit(X_train_selected, y_train)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test_selected)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy on the test set: {accuracy:.2f}')

# Get the selected feature indices
selected_feature_indices = np.where(rfe_selector.support_)[0]
print('Selected feature indices:', selected_feature_indices)
# Get the names of the selected features
selected_feature_names = X.columns[selected_feature_indices]  # Assuming X is a DataFrame
print('Selected feature names:', selected_feature_names)

#%%
''' 
Forward Stepwise Selection is a wrapper method where features are added one at a 
time to the model until a certain criterion is met. In each step, the feature that 
provides the best improvement in the chosen criterion is added to the set of selected
 features. Here's an example of using Forward Stepwise Selection for feature selection
 with an SVM classifier:
'''
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SequentialFeatureSelector

# Assuming you have your data (X, y) loaded

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features (important for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create an SVM classifier
svm_classifier = SVC(kernel='rbf')  # You can specify the kernel you want to use

# Use Forward Stepwise Selection for feature selection
num_features_to_select = 5  # Specify the number of features you want to select
fss_selector = SequentialFeatureSelector(svm_classifier, n_features_to_select=num_features_to_select, direction='forward')
X_train_selected = fss_selector.fit_transform(X_train_scaled, y_train)
X_test_selected = fss_selector.transform(X_test_scaled)

# Train SVM classifier on the selected features
svm_classifier.fit(X_train_selected, y_train)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test_selected)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy on the test set: {accuracy:.2f}')

# Get the selected feature indices
selected_feature_indices = fss_selector.get_support(indices=True)
print('Selected feature indices:', selected_feature_indices)

# Get the names of the selected features
selected_feature_names = X.columns[selected_feature_indices]  # Assuming X is a DataFrame
print('Selected feature names:', selected_feature_names)

#%%  Forward Stepwise Selection in Grid Search.
#https://www.analyticsvidhya.com/blog/2016/12/introduction-to-feature-selection-methods-with-an-example-or-how-to-select-the-right-variables/
 
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.pipeline import Pipeline

# Assuming you have your data (X, y) loaded

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features (important for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create an SVM classifier
svm_classifier = SVC()

# Use Forward Stepwise Selection for feature selection
num_features_to_select = 10  # Specify the number of features you want to select
#fss_selector = SequentialFeatureSelector(svm_classifier, n_features_to_select=num_features_to_select, direction='forward')

rfe_selector = RFE(svm_classifier, n_features_to_select=num_features_to_select)


# Set up the parameter grid for GridSearchCV
param_grid = {
    'selector__estimator__C': [0.1],  # SVM hyperparameters
    'selector__estimator__kernel': ['linear'],  # SVM kernel options
}

# Create a pipeline with Forward Stepwise Selection and SVM classifier
pipeline = Pipeline([
    #('selector', fss_selector),
    ('selector',rfe_selector),
    ('estimator', svm_classifier),
])

# Initialize GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5)  # Adjust the number of folds as needed

# Fit the grid search to the training data
grid_search.fit(X_train_scaled, y_train)

# Print the best parameters and corresponding accuracy
print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)

# Evaluate the performance on the test set using the best model from grid search
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test_scaled)
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"Accuracy on the test set with the best model: {accuracy_best}")

# Get the selected feature indices
selected_feature_indices = best_model.named_steps['selector'].get_support(indices=True)
print('Selected feature indices:', selected_feature_indices)

# Get the names of the selected features
selected_feature_names = X.columns[selected_feature_indices]  # Assuming X is a DataFrame
print('Selected feature names:', selected_feature_names)



#%%https://www.visual-design.net/post/feature-selection-and-eda-in-machine-learning
'''
Embedded methods for feature selection involve incorporating the feature selection process 
directly into the model training. One common embedded method is LASSO (L1 Regularization)
for linear models. For SVM, you can use linear SVM with L1 regularization. 
Here's an example using linear SVM with L1 regularization for feature selection:
'''
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Assuming you have your data (X, y) loaded

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features (important for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a linear SVM classifier with L1 regularization
svm_classifier = LinearSVC(penalty='l1', dual=False, max_iter=10000)


# Train SVM classifier with feature selection
svm_classifier.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test_scaled)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy on the test set: {accuracy:.2f}')

# Get the selected feature indices
selected_feature_indices = np.where(svm_classifier.coef_ != 0)[1]
print('Selected feature indices:', selected_feature_indices)

#%%
'''

Embedded methods perform feature selection as part of the model training process. 
For SVM, linear models with built-in feature selection capabilities, such as LinearSVC
with L1 regularization, are often used as embedded methods.  Another popular embedded 
method is tree-based feature importances, which can be applied to SVM with a 
linear kernel. Here's an example using tree-based feature importances:
'''

'''
Can I use Random forests for feature selection and then use SVM for classification?

As Meir Maor mentioned, of course you can. However, when we perform feature selection with
one algorithm and classifies with another, we should always be mindful of how each
algorithm works, especially in the assumptions it makes.Random forests, through their 
base classifier, make the assumption that splitting the the input space into 
hyperrectangles makes sense in terms of classification. SVMs project the data 
into a higher dimensional space so they can be separated linearly in that space. 
Unless we have one very interesting kernel in our SVM, it's unlikely that the two 
will have any sort of equivalence in the importance of variables.
That's not to say that it won't work in a given scenario or for a specific type 
of problem, but to expect it to work generally is probably not a good idea.
'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Assuming you have your data (X, y) loaded

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features (important for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a RandomForestClassifier to estimate feature importances
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train RandomForestClassifier on the entire dataset
rf_classifier.fit(X_train_scaled, y_train)

# Get feature importances
feature_importances = rf_classifier.feature_importances_

# Select top k features based on importance
num_features_to_select = 10  # Specify the number of features you want to select
top_k_indices = feature_importances.argsort()[-num_features_to_select:][::-1]

# Get the names of the selected features
selected_feature_names = X.columns[top_k_indices]  # Assuming X is a DataFrame

# Print the names of the selected features
print('Selected feature names with Embedded Method:', selected_feature_names)

# Train SVM classifier on the selected features
svm_classifier = LinearSVC()
svm_classifier.fit(X_train_scaled[:, top_k_indices], y_train)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test_scaled[:, top_k_indices])

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy on the test set with Embedded Method: {accuracy:.2f}')

#%%

#%% RFE logistic regression

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SequentialFeatureSelector

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features (important for logistic regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a logistic regression classifier
logreg_classifier = LogisticRegression(max_iter=1000)

# Apply RFE to the logistic regression classifier
num_features_to_select = 10  # Specify the number of features you want to select
# rfe_selector = RFE(logreg_classifier, n_features_to_select=num_features_to_select)
# X_train_selected = rfe_selector.fit_transform(X_train_scaled, y_train)
# X_test_selected = rfe_selector.transform(X_test_scaled)

# # Use Forward Stepwise Selection for feature selection
fss_selector = SequentialFeatureSelector(logreg_classifier, n_features_to_select=num_features_to_select, direction='forward')
X_train_selected = fss_selector.fit_transform(X_train_scaled, y_train)
X_test_selected = fss_selector.transform(X_test_scaled)

# Train logistic regression classifier on the selected features
logreg_classifier.fit(X_train_selected, y_train)

# Make predictions on the test set
y_pred = logreg_classifier.predict(X_test_selected)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy on the test set: {accuracy:.2f}')

# Get the selected feature indices
selected_feature_indices = np.where(rfe_selector.support_)[0]
print('Selected feature indices:', selected_feature_indices)
# Get the names of the selected features
selected_feature_names = X.columns[selected_feature_indices]  # Assuming X is a DataFrame
print('Selected feature names:', selected_feature_names)





# # Train SVM classifier on the selected features
# svm_classifier.fit(X_train_selected, y_train)

# # Make predictions on the test set
# y_pred = svm_classifier.predict(X_test_selected)


# # Calculate and plot the confusion matrix
# conf_matrix = confusion_matrix(y_test, y_pred)

# # Normalize the confusion matrix
# conf_matrix_normalized = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)

# # Plot normalized confusion matrix with a larger figure size
# plt.figure(figsize=(10, 8))
# sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2%', cmap='Blues', cbar=True, square=True,
#             xticklabels=years, yticklabels=years, annot_kws={"size": 12})

# plt.title('Normalized Confusion Matrix (Logistic Regression)')
# plt.xlabel('Predicted Year')
# plt.ylabel('True Year')
# plt.show()

#%%logistic: Gridsearch
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Assuming you have your data (X, y) loaded
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features (important for logistic regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a logistic regression classifier
logreg_classifier = LogisticRegression()

# Set up the parameter grid for GridSearchCV
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2', 'elasticnet'],
    'solver': ['liblinear', 'lbfgs', 'newton-cg', 'sag'],
    'max_iter': [100, 200, 300],
    'class_weight': [None, 'balanced']
}

# Initialize GridSearchCV
grid_search = GridSearchCV(logreg_classifier, param_grid, cv=5)

# Fit the grid search to the training data
grid_search.fit(X_train_scaled, y_train)

# Print the best parameters and corresponding accuracy
print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)


#%% Naive Bayes with wrapper method not working
''' The error you are encountering suggests that GaussianNB does not have 
the coef_ or feature_importances_ attribute, which is required by the RFE 
feature selection method with the default importance_getter set to 'auto'.'''

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np

from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Assuming you have your data (X, y) loaded

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features (important for some algorithms)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a Naive Bayes classifier (Gaussian Naive Bayes for continuous features)
naive_bayes_classifier = GaussianNB()

# Apply RFE to the Naive Bayes classifier
num_features_to_select = 10  # Specify the number of features you want to select
rfe_selector = RFE(naive_bayes_classifier, n_features_to_select=num_features_to_select)
X_train_selected = rfe_selector.fit_transform(X_train_scaled, y_train)
X_test_selected = rfe_selector.transform(X_test_scaled)


# Train Naive Bayes classifier on the selected features
naive_bayes_classifier.fit(X_train_selected, y_train)

# Make predictions on the test set
y_pred = naive_bayes_classifier.predict(X_test_selected)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy on the test set: {accuracy:.2f}')

# Get the selected feature indices
selected_feature_indices = np.where(rfe_selector.support_)[0]
print('Selected feature indices:', selected_feature_indices)

# Get the names of the selected features
selected_feature_names = X.columns[selected_feature_indices]  # Assuming X is a DataFrame
print('Selected feature names:', selected_feature_names)

#%% Naive Bayes

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from mlxtend.feature_selection import SequentialFeatureSelector
import numpy as np

# Assuming you have your data (X, y) loaded

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features (important for some algorithms)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a Naive Bayes classifier (Gaussian Naive Bayes for continuous features)
naive_bayes_classifier = GaussianNB()

# Apply RFE from mlxtend to the Naive Bayes classifier
num_features_to_select = 10  # Specify the number of features you want to select
sfs_selector = SequentialFeatureSelector(naive_bayes_classifier, 
                                         k_features=num_features_to_select, 
                                         forward=True, floating=False, verbose=2, 
                                         scoring='accuracy', cv=5)
X_train_selected = sfs_selector.fit_transform(X_train_scaled, y_train)
X_test_selected = sfs_selector.transform(X_test_scaled)

# Train Naive Bayes classifier on the selected features
naive_bayes_classifier.fit(X_train_selected, y_train)

# Make predictions on the test set
y_pred = naive_bayes_classifier.predict(X_test_selected)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy on the test set: {accuracy:.2f}')

# Get the selected feature indices
selected_feature_indices = np.array(sfs_selector.k_feature_idx_)
print('Selected feature indices:', selected_feature_indices)

# Get the names of the selected features
selected_feature_names = X.columns[selected_feature_indices]  # Assuming X is a DataFrame
print('Selected feature names:', selected_feature_names)

#%%


#%%  Naive Bayes with filter method
''' For the Gaussian Naive Bayes classifier (GaussianNB) in scikit-learn, 
there are no hyperparameters that can be explicitly tuned using traditional 
methods like grid search or cross-validation because it assumes that all features 
are independent and follows a Gaussian distribution. As a result, there are no 
hyperparameters like those found in more complex models.'''

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np

# Assuming you have your data (X, y) loaded

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features (important for some algorithms)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a Naive Bayes classifier (Gaussian Naive Bayes for continuous features)
naive_bayes_classifier = GaussianNB()

# Apply SelectKBest with mutual information to the Naive Bayes classifier
num_features_to_select = 10  # Specify the number of features you want to select
#selector = SelectKBest(mutual_info_classif, k=num_features_to_select)
selector = SelectKBest(f_classif, k=num_features_to_select)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

# Train Naive Bayes classifier on the selected features
naive_bayes_classifier.fit(X_train_selected, y_train)

# Make predictions on the test set
y_pred = naive_bayes_classifier.predict(X_test_selected)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy on the test set: {accuracy:.2f}')

# Get the selected feature indices
selected_feature_indices = np.where(selector.get_support())[0]
print('Selected feature indices:', selected_feature_indices)

# Get the names of the selected features
selected_feature_names = X.columns[selected_feature_indices]  # Assuming X is a DataFrame
print('Selected feature names:', selected_feature_names)


#%% KNN with mutual_infi_classif

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Assuming you have your data (X, y) loaded

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features (important for KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SelectKBest with mutual information to the data
num_features_to_select = 10  # Specify the number of features you want to select
selector = SelectKBest(mutual_info_classif, k=num_features_to_select)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

# Create a KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=6)  # You can adjust the number of neighbors

# Train KNN classifier on the selected features
knn_classifier.fit(X_train_selected, y_train)

# Make predictions on the test set
y_pred = knn_classifier.predict(X_test_selected)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy on the test set: {accuracy:.2f}')

# Get the selected feature indices
selected_feature_indices = np.where(selector.get_support())[0]
print('Selected feature indices:', selected_feature_indices)

# Get the names of the selected features
selected_feature_names = X.columns[selected_feature_indices]  # Assuming X is a DataFrame
print('Selected feature names:', selected_feature_names)

#%% KNN using sklearn forward selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SequentialFeatureSelector
import numpy as np

# Assuming you have your data (X, y) loaded

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features (important for KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors

# Use Sequential Feature Selector for feature selection
num_features_to_select = 10  # Specify the number of features you want to select
sfs_selector = SequentialFeatureSelector(knn_classifier, 
                                         n_features_to_select=num_features_to_select, 
                                         direction='forward')

# Fit the selector on training data
sfs_selector.fit(X_train_scaled, y_train)

# Get the selected feature indices
selected_feature_indices = np.where(sfs_selector.get_support())[0]
print('Selected feature indices:', selected_feature_indices)

# Get the names of the selected features
selected_feature_names = X.columns[selected_feature_indices]  # Assuming X is a DataFrame
print('Selected feature names:', selected_feature_names)

# Train KNN classifier on the selected features
knn_classifier.fit(X_train_scaled[:, selected_feature_indices], y_train)

# Make predictions on the test set
y_pred = knn_classifier.predict(X_test_scaled[:, selected_feature_indices])

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy on the test set: {accuracy:.2f}')

#%% KNN with SequentialFeatureSelector mlxtend
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from mlxtend.feature_selection import SequentialFeatureSelector
import numpy as np
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
# Assuming you have your data (X, y) loaded

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features (important for KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=6)  # You can adjust the number of neighbors

# Use Sequential Feature Selector for feature selection
num_features_to_select = 10  # Specify the number of features you want to select
sfs_selector = SequentialFeatureSelector(knn_classifier,
                                         k_features=num_features_to_select, 
                                         forward=True, floating=False, verbose=2,
                                         scoring='accuracy',
                                         cv=0)
sfs_selector.fit(X_train_scaled, y_train)

# Get the selected feature indices
selected_feature_indices = list(sfs_selector.k_feature_idx_)
print('Selected feature indices:', selected_feature_indices)

# Get the names of the selected features
selected_feature_names = X.columns[selected_feature_indices]  # Assuming X is a DataFrame
print('Selected feature names:', selected_feature_names)

# Train KNN classifier on the selected features
knn_classifier.fit(X_train_scaled[:, selected_feature_indices], y_train)

# Make predictions on the test set
y_pred = knn_classifier.predict(X_test_scaled[:, selected_feature_indices])

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy on the test set: {accuracy:.2f}')

# Plot the results
fig1 = plot_sfs(sfs_selector.get_metric_dict(), kind='std_dev')

plt.ylim([0.8, 1])
plt.title('Sequential Forward Selection (w. StdDev)')
plt.grid()
plt.show()

#%% KNN RFE doesnt work
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE
import numpy as np

# Assuming you have your data (X, y) loaded

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features (important for KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors

# Use Recursive Feature Elimination (RFE) for feature selection
num_features_to_select = 10  # Specify the number of features you want to select
rfe_selector = RFE(knn_classifier, n_features_to_select=num_features_to_select)

# Fit the selector on training data
rfe_selector = rfe_selector.fit(X_train_scaled, y_train)

# Get the selected feature indices
selected_feature_indices = np.where(rfe_selector.support_)[0]
print('Selected feature indices:', selected_feature_indices)

# Get the names of the selected features
selected_feature_names = X.columns[selected_feature_indices]  # Assuming X is a DataFrame
print('Selected feature names:', selected_feature_names)

# Train KNN classifier on the selected features
knn_classifier.fit(X_train_scaled[:, selected_feature_indices], y_train)

# Make predictions on the test set
y_pred = knn_classifier.predict(X_test_scaled[:, selected_feature_indices])

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy on the test set: {accuracy:.2f}')

#%%
#%%
#%%
#%%

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Assuming you have feature_selection_matrix, a binary matrix indicating selected features
# Replace ... with your data
feature_selection_matrix = pd.DataFrame(np.random.randint(2, size=(20, 5)), columns=['Iter1', 'Iter2', 'Iter3', 'Iter4', 'Iter5'])

plt.figure(figsize=(10, 6))
sns.heatmap(feature_selection_matrix, cmap="Blues", cbar=False, annot=False)
plt.xlabel('Iteration')
plt.ylabel('Feature Index')
plt.title('Feature Selection Matrix')
plt.show()


#


#%%
#%% 
#%%
#%%

#Classifiers in their simplest form
#%% decision tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Assuming X is your feature matrix and y is the target variable (year)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train a decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Make predictions on the test set
predictions = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

#%% DecisionTreeClassifier with RFE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE
import numpy as np

# Assuming you have your data (X, y) loaded

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features (important for Decision Tree as well)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a Decision Tree classifier
dt_classifier = DecisionTreeClassifier(random_state=42)

# Use Recursive Feature Elimination (RFE) for feature selection
num_features_to_select = 10  # Specify the number of features you want to select
rfe_selector = RFE(dt_classifier, n_features_to_select=num_features_to_select)

# Fit the selector on training data
rfe_selector = rfe_selector.fit(X_train_scaled, y_train)

# Get the selected feature indices
selected_feature_indices = np.where(rfe_selector.support_)[0]
print('Selected feature indices:', selected_feature_indices)

# Get the names of the selected features
selected_feature_names = X.columns[selected_feature_indices]  # Assuming X is a DataFrame
print('Selected feature names:', selected_feature_names)

# Train Decision Tree classifier on the selected features
dt_classifier.fit(X_train_scaled[:, selected_feature_indices], y_train)

# Make predictions on the test set
y_pred = dt_classifier.predict(X_test_scaled[:, selected_feature_indices])

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy on the test set: {accuracy:.2f}')


#%% DecisionTreeClassifier with sFS

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SequentialFeatureSelector
import numpy as np

# Assuming you have your data (X, y) loaded

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features (important for Decision Tree as well)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a Decision Tree classifier
dt_classifier = DecisionTreeClassifier(random_state=42)

# Use Sequential Feature Selector for feature selection
num_features_to_select = 10  # Specify the number of features you want to select
sfs_selector = SequentialFeatureSelector(dt_classifier, 
                                         n_features_to_select=num_features_to_select, 
                                         direction='forward')

# Fit the selector on training data
sfs_selector.fit(X_train_scaled, y_train)

# Get the selected feature indices
selected_feature_indices = np.where(sfs_selector.get_support())[0]
print('Selected feature indices:', selected_feature_indices)

# Get the names of the selected features
selected_feature_names = X.columns[selected_feature_indices]  # Assuming X is a DataFrame
print('Selected feature names:', selected_feature_names)

# Train Decision Tree classifier on the selected features
dt_classifier.fit(X_train_scaled[:, selected_feature_indices], y_train)

# Make predictions on the test set
y_pred = dt_classifier.predict(X_test_scaled[:, selected_feature_indices])

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy on the test set: {accuracy:.2f}')

#%% DecisionTreeClassifier with embedded feature_importances_

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Assuming you have your data (X, y) loaded

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features (important for Decision Tree as well)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a Decision Tree classifier
dt_classifier = DecisionTreeClassifier(random_state=42)

# Train Decision Tree classifier
dt_classifier.fit(X_train_scaled, y_train)

# Get feature importances
feature_importances = dt_classifier.feature_importances_

# Sort indices based on feature importance
sorted_feature_indices = np.argsort(feature_importances)[::-1]

# Select the top num_features_to_select features
num_features_to_select = 10  # Specify the number of features you want to select
selected_feature_indices = sorted_feature_indices[:num_features_to_select]
print('Selected feature indices:', selected_feature_indices)

# Get the names of the selected features
selected_feature_names = X.columns[selected_feature_indices]  # Assuming X is a DataFrame
print('Selected feature names:', selected_feature_names)

# Train Decision Tree classifier on the selected features
dt_classifier.fit(X_train_scaled[:, selected_feature_indices], y_train)

# Make predictions on the test set
y_pred = dt_classifier.predict(X_test_scaled[:, selected_feature_indices])

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy on the test set: {accuracy:.2f}')

import matplotlib.pyplot as plt

# Plot the feature importance values
plt.figure(figsize=(10, 6))
plt.bar(range(num_features_to_select), feature_importances[selected_feature_indices], align='center')
plt.xticks(range(num_features_to_select), selected_feature_names, rotation=45)
plt.xlabel('Feature Names')
plt.ylabel('Feature Importance')
plt.title('Feature Importance of Decision Tree classifier')
plt.tight_layout()
plt.show()


#%% DecisionTreeClassifier hyperparameter tuning
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Assuming you have your data (X, y) loaded
# Assuming you have 'selected_feature_indices' obtained from Sequential Feature Selector or other method

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features (important for Decision Tree as well)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a Decision Tree classifier
dt_classifier = DecisionTreeClassifier(random_state=42)

# Fit the classifier on training data
dt_classifier.fit(X_train_scaled[:, selected_feature_indices], y_train)

# Get the selected feature importances
feature_importances = dt_classifier.feature_importances_

# Print feature importances
print('Feature Importances:', feature_importances)

# Define the parameter grid
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(dt_classifier, param_grid, cv=5)
grid_search.fit(X_train_scaled[:, selected_feature_indices], y_train)

# Get the best parameters
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Train Decision Tree classifier with the best parameters
best_dt_classifier = DecisionTreeClassifier(random_state=42, **best_params)
best_dt_classifier.fit(X_train_scaled[:, selected_feature_indices], y_train)

# Make predictions on the test set
y_pred_best = best_dt_classifier.predict(X_test_scaled[:, selected_feature_indices])

# Evaluate performance
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f'Accuracy on the test set with the best model: {accuracy_best:.2f}')


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
    shannon_index = -sum(p * math.log2(p) if p != 0 else 0 for p in relative_abundance)

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

# Calculate correlation matrix
correlation_matrix = X.corr()

# Plot correlation matrix as a heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix')
plt.show()

#%% Feature Selection RandomForestClassifier
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Assuming X and y are your feature matrix and target variable

# Create a Random Forest classifier
random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
#random_forest_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Fit the model to the data
random_forest_classifier.fit(X, y)

# Get feature importances
feature_importances = random_forest_classifier.feature_importances_

# Create a DataFrame to display feature importances
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

# Sort the DataFrame by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Select the top k features based on importance
k = 10  # Choose the desired number of features
selected_features = feature_importance_df['Feature'].head(k).tolist()

# Use only the selected features
X_selected = X[selected_features]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Train the model on the selected features
random_forest_classifier_selected = RandomForestClassifier(n_estimators=100, random_state=42)
#random_forest_classifier_selected = GradientBoostingClassifier(n_estimators=100, random_state=42)
random_forest_classifier_selected.fit(X_train, y_train)

# Make predictions on the test set
predictions = random_forest_classifier_selected.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")



#%% Feature Selection GradientBoostingClassifier
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Assuming X and y are your feature matrix and target variable

# Create a Random Forest classifier
gradient_boosting_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Fit the model to the data
gradient_boosting_classifier.fit(X, y)

# Get feature importances
feature_importances = random_forest_classifier.feature_importances_

# Create a DataFrame to display feature importances
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

# Sort the DataFrame by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Select the top k features based on importance
k = 10  # Choose the desired number of features
selected_features = feature_importance_df['Feature'].head(k).tolist()

# Use only the selected features
X_selected = X[selected_features]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Train the model on the selected features
gradient_boosting_classifier_selected = GradientBoostingClassifier(n_estimators=100, random_state=42)
gradient_boosting_classifier_selected.fit(X_train, y_train)

# Make predictions on the test set
predictions = gradient_boosting_classifier_selected.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

#%% GradientBoostingClassifier:SequentialFeatureSelector:mlxtend

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
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

# Create a GradientBoosting classifier
gradient_boosting_classifier = GradientBoostingClassifier(random_state=42)

# Apply sequential forward feature selection from mlxtend to the GradientBoosting classifier
num_features_to_select = 10  # Specify the number of features you want to select
sfs_selector = SequentialFeatureSelector(gradient_boosting_classifier, 
                                         k_features=num_features_to_select, 
                                         forward=True, floating=False, verbose=2, 
                                         scoring='accuracy', cv=5)
X_train_selected = sfs_selector.fit_transform(X_train_scaled, y_train)
X_test_selected = sfs_selector.transform(X_test_scaled)

# Train GradientBoosting classifier on the selected features
gradient_boosting_classifier.fit(X_train_selected, y_train)

# Make predictions on the test set
y_pred = gradient_boosting_classifier.predict(X_test_selected)

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

plt.ylim([0.8, 1])
plt.title('Sequential Forward Selection (GradientBoosting Classifier)')
plt.grid()
plt.show()

#%% Model saving


from joblib import dump, load
import os

# Specify the directory where you want to save the model
save_directory = 'S:/mc5545/SA_Drone_data/trained_classifiers/'

# Create the directory if it doesn't exist
os.makedirs(save_directory, exist_ok=True)

# Save the trained Random Forest classifier in the specified directory
model_filename = 'random_forest_model.joblib'
model_filepath = os.path.join(save_directory, model_filename)
dump(gradient_boosting_classifier, model_filepath)

# Print the full path to the saved model
print(f'Model saved to: {model_filepath}')

#%% MLPClassifier
'''
For the parameter hidden_layer_sizes=(100, 50), it indicates a neural network 
with two hidden layers. The first hidden layer has 100 neurons, and the second 
hidden layer has 50 neurons. You can customize the architecture of the neural 
network by adjusting the values in the tuple.'''

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from mlxtend.feature_selection import SequentialFeatureSelector
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import numpy as np

# Assuming you have your data (X, y) loaded

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features (important for MLP)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create an MLP classifier
mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,50), max_iter=1000, random_state=42)
# You can adjust the hidden_layer_sizes, max_iter, and other hyperparameters based on your data

# Use Sequential Feature Selector (SFS) for feature selection
num_features_to_select = 10  # Specify the number of features you want to select
sfs_selector = SequentialFeatureSelector(mlp_classifier, 
                                         k_features=num_features_to_select, 
                                         forward=True, floating=False, verbose=2,
                                         scoring='accuracy', cv=0)
sfs_selector.fit(X_train_scaled, y_train)

# Get the selected feature indices
selected_feature_indices = list(sfs_selector.k_feature_idx_)
print('Selected feature indices:', selected_feature_indices)

# Get the names of the selected features
selected_feature_names = X.columns[selected_feature_indices]  # Assuming X is a DataFrame
print('Selected feature names:', selected_feature_names)

# Train MLP classifier on the selected features
mlp_classifier.fit(X_train_scaled[:, selected_feature_indices], y_train)

# Make predictions on the test set
y_pred = mlp_classifier.predict(X_test_scaled[:, selected_feature_indices])

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy on the test set: {accuracy:.2f}')

#%% filter methods: MLP
'''
hidden_layer_sizes : This parameter allows us to set the number of layers and the number 
of nodes we wish to have in the Neural Network Classifier. Each element in the tuple 
represents the number of nodes at the ith position where i is the index of the tuple. 
Thus the length of tuple denotes the total number of hidden layers in the network.
max_iter: It denotes the number of epochs.
activation: The activation function for the hidden layers.
solver: This parameter specifies the algorithm for weight optimization across the nodes.
random_state: The parameter allows to set a seed for reproducing the same results
After initializing we can now give the data to train the Neural Network.
'''
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Assuming you have your data (X, y) loaded
# X should be a 2D array-like structure (samples, features)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features (important for neural networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# Create an MLP classifier
#mlp_classifier = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=1000)
#Initializing the MLPClassifier
classifier = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=500,
                           activation = 'relu',solver='adam',random_state=42)

# Use SelectKBest for feature selection
num_features_to_select = 10  # Specify the number of features you want to select
feature_selector = SelectKBest(f_classif, k=num_features_to_select)
X_train_selected = feature_selector.fit_transform(X_train_scaled, y_train)
X_test_selected = feature_selector.transform(X_test_scaled)

# Get the selected feature indices
selected_feature_indices = feature_selector.get_support(indices=True)

# Get the names of the selected features
selected_feature_names = X.columns[selected_feature_indices]  # Assuming X is a DataFrame

# Print the selected feature names
print('Selected feature names:', selected_feature_names)

# Train MLP classifier on the selected features
mlp_classifier.fit(X_train_selected, y_train)

# Make predictions on the test set
y_pred = mlp_classifier.predict(X_test_selected)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy on the test set: {accuracy:.2f}')

#%% MLP: SFS:Forward
#RFE doesn't work
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import accuracy_score

# Assuming you have your data (X, y) loaded
# X should be a 2D array-like structure (samples, features)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features (important for neural networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create an MLP classifier
mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,50), max_iter=100)

# Apply RFE to the MLP classifier
num_features_to_select = 10  # Specify the number of features you want to select
sfs_selector = SequentialFeatureSelector(mlp_classifier, 
                                         n_features_to_select=num_features_to_select,
                                         direction='forward')

# Fit the selector on training data
sfs_selector.fit(X_train_scaled, y_train)

# Get the selected feature indices
selected_feature_indices = np.where(sfs_selector.get_support())[0]
print('Selected feature indices:', selected_feature_indices)

# Get the names of the selected features
selected_feature_names = X.columns[selected_feature_indices]  # Assuming X is a DataFrame
print('Selected feature names:', selected_feature_names)

# Train MLP classifier on the selected features
mlp_classifier.fit(X_train_scaled[:, selected_feature_indices], y_train)

# Make predictions on the test set
y_pred = mlp_classifier.predict(X_test_scaled[:, selected_feature_indices])

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy on the test set: {accuracy:.2f}')


#%% LSTM

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Assuming you have your data (X, y) loaded
# X should be in the shape (samples, time_steps, features)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features (important for neural networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Design the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])))
model.add(Dense(1, activation='sigmoid'))  # Assuming binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the model on the test set
y_pred_proba = model.predict(X_test_scaled)
y_pred = (y_pred_proba > 0.5).astype(int)  # Assuming binary classification

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy on the test set: {accuracy:.2f}')



#%% LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming X and y are your feature matrix and target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (important for logistic regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a Logistic Regression model with L1 regularization
logistic_regression_model = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=42)

# Train the model on the training set
logistic_regression_model.fit(X_train, y_train)

# Make predictions on the test set
predictions = logistic_regression_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

# Calculate and plot the confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)

# Normalize the confusion matrix
conf_matrix_normalized = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)

# Plot normalized confusion matrix with a larger figure size
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2%', cmap='Blues', cbar=True, square=True,
            xticklabels=logistic_regression_model.classes_, yticklabels=logistic_regression_model.classes_,
            annot_kws={"size": 12})

plt.title('Normalized Confusion Matrix (Logistic Regression)')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Assuming X and y are your feature matrix and target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (important for logistic regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define a range of C values to experiment with
c_values = [0.001, 0.01, 0.1, 1, 10, 100]

# Lists to store accuracy scores for different C values
train_accuracy = []
test_accuracy = []

# Iterate over different C values
for c_value in c_values:
    # Create a Logistic Regression model with L1 regularization
    logistic_regression_model = LogisticRegression(penalty='l1', solver='liblinear', C=c_value, random_state=42)
    
    # Train the model on the training set
    logistic_regression_model.fit(X_train, y_train)
    
    # Make predictions on the training set
    train_predictions = logistic_regression_model.predict(X_train)
    
    # Make predictions on the test set
    test_predictions = logistic_regression_model.predict(X_test)
    
    # Calculate accuracy scores and store them
    train_accuracy.append(accuracy_score(y_train, train_predictions))
    test_accuracy.append(accuracy_score(y_test, test_predictions))

# Plot the effect of different C values on accuracy
plt.figure(figsize=(10, 6))
plt.plot(np.log10(c_values), train_accuracy, label='Training Accuracy')
plt.plot(np.log10(c_values), test_accuracy, label='Test Accuracy')
plt.title('Effect of L1 Regularization (Logistic Regression)')
plt.xlabel('log10(C)')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Assuming X and y are your feature matrix and target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (important for logistic regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a Logistic Regression model with L1 regularization
logistic_regression_model = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=42)

# Train the model on the training set
logistic_regression_model.fit(X_train, y_train)

# Get selected features and their coefficients
selected_features = X.columns[logistic_regression_model.coef_[0] != 0]
coefficients = logistic_regression_model.coef_[0][logistic_regression_model.coef_[0] != 0]

# Create a DataFrame to display selected features and coefficients
selected_features_df = pd.DataFrame({'Feature': selected_features, 'Coefficient': coefficients})

# Sort the DataFrame by absolute coefficient values in descending order
selected_features_df = selected_features_df.reindex(selected_features_df['Coefficient'].abs().sort_values(ascending=False).index)

# Display the DataFrame
print(selected_features_df)


#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming X and y are your feature matrix and target variable


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (important for linear models)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a Support Vector Classifier (SVC) with linear kernel and L1 regularization
linear_svc_classifier = SVC(kernel='linear', C=1.0, penalty='l1', dual=False, random_state=42)

# Train the model on the training set
linear_svc_classifier.fit(X_train, y_train)

# Make predictions on the test set
predictions = linear_svc_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

# Calculate and plot the confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)

# Normalize the confusion matrix
conf_matrix_normalized = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)

# Plot normalized confusion matrix with a larger figure size
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2%', cmap='Blues', cbar=True, square=True,
            xticklabels=linear_svc_classifier.classes_, yticklabels=linear_svc_classifier.classes_,
            annot_kws={"size": 12})

plt.title('Normalized Confusion Matrix (LinearSVC with L1 Regularization)')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.show()

#https://www.shedloadofcode.com/blog/eight-ways-to-perform-feature-selection-with-scikit-learn

#%%%
#%%
#%%
#%%
#Predicting alpha diversity 
'''
Given that you have a single alpha diversity value, we can still estimate alpha diversity 
for all coordinates using the available data. Here’s an alternative approach:

Create a Similarity Metric:
1. Calculate a similarity metric (e.g., Euclidean distance, cosine similarity) between 
the features of each sample and the features of the sample with the known alpha diversity.
This similarity metric will help us identify samples that are most similar to the known 
alpha diversity sample.
2. Weighted Average:
Assign weights to each sample based on their similarity to the known alpha diversity 
sample.The more similar a sample’s features are to the known sample, the higher its 
weight.Compute a weighted average of the alpha diversities of all samples using these 
weights.
3. Predict Alpha Diversity:
Use the weighted average to estimate alpha diversity for all coordinates.
Below is a Python code snippet demonstrating this approach. Note that you’ll need to 
replace the example data with your actual dataset:
'''

import numpy as np
from scipy.spatial.distance import euclidean

# Example data: x, y coordinates, 10 features, and one known alpha diversity
x_coords = np.array([1, 2, 3, 4, 5])
y_coords = np.array([10, 20, 30, 40, 50])
features = np.random.rand(5, 10)  # 5 samples, 10 features
known_alpha_diversity = 0.7

# Calculate similarity metric (Euclidean distance) between features
distances = np.array([euclidean(features[i], features[0]) for i in range(len(features))])

# Compute weights based on inverse distances
weights = 1 / (1 + distances)

# Weighted average of alpha diversities
estimated_alpha_diversity = np.sum(weights * known_alpha_diversity) / np.sum(weights)

print(f"Estimated Alpha Diversity for all coordinates: {estimated_alpha_diversity:.4f}")

#%%
import numpy as np
from scipy.spatial.distance import euclidean

# Example data: x, y coordinates, 10 features, and one known alpha diversity
x_coords = np.array([1, 2, 3, 4, 5])
y_coords = np.array([10, 20, 30, 40, 50])
features = np.random.rand(5, 10)  # 5 samples, 10 features
known_alpha_diversity = 0.7

# Calculate similarity metric (Euclidean distance) between features and coordinates
distances = np.array([euclidean(features[i], features[0]) + 
                      euclidean([x_coords[i], y_coords[i]], [x_coords[0], y_coords[0]]) 
                      for i in range(len(features))])

# Compute weights based on inverse distances
weights = 1 / (1 + distances)

# Calculate individual alpha diversities for each coordinate
individual_alpha_diversities = weights * known_alpha_diversity

# Print the individual alpha diversities for each coordinate
for i, (x, y) in enumerate(zip(x_coords, y_coords)):
    print(f"Coordinate ({x}, {y}): Individual Alpha Diversity = {individual_alpha_diversities[i]:.4f}")
