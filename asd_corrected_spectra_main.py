# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 21:41:21 2023

@author: Manisha Das
"""
# #https://tothepoles.co.uk/2018/04/25/asd-spectra-processing-with-linux-python/


#%% ASD file read and jump correction

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os



def read_asd_file(file_path):
    wavelength_data = []
    reflectance = []

    with open(file_path, 'r') as file:
        found_data = False
        filename = os.path.basename(file_path)  # Get the filename from the file path
        filename = os.path.splitext(filename)[0]
        for line in file:
            if found_data:
                # Split the line into wavelength and asd values
                data = line.strip().split(',')
                if len(data) == 2:
                    wavelength, asd_value = data
                    wavelength_data.append(int(wavelength))
                    reflectance.append(float(asd_value))
            elif line.strip().startswith(f'Wavelength,{filename}'):
                found_data = True

    # Convert the lists to NumPy arrays
    wavelength_data = np.array(wavelength_data)
    reflectance = np.array(reflectance)

    return wavelength_data, reflectance


#
# Operator.py defines operations on pd.Series that consists of
# wavelength as index and measurement as values
# jump_correct: resolve jumps in non-overlapping wavelengths
def jump_correct(series, splices, reference, method="additive"):
    """
    Correct for jumps in non-overlapping wavelengths
    
    Parameters
    ----------
    splices: list
        list of wavelength values where jumps occur
    
    reference: int
        position of the reference band (0-based)
    
    """
    if method == "additive":
        return jump_correct_additive(series, splices, reference)

def jump_correct_additive(series, splices, reference):
    """ Perform additive jump correction (ASD) """
    # if asd, get the locations from the metadata
    # stop if overlap exists
    def get_sequence_num(wavelength):
        """ return the sequence id after cutting at splices """
        for i, splice in enumerate(splices):
            if wavelength <= splice:
                return i
        return i+1
    def translate_y(ref, mov, right=True):
        # translates the mov sequence to stitch with ref sequence
        if right:
            diff = ref.iloc[-1] - mov.iloc[0]
        else:
            diff = ref.iloc[0] - mov.iloc[-1]
        mov = mov + diff
        series.update(mov)
    groups = series.groupby(get_sequence_num)
    for i in range(reference, groups.ngroups-1, 1):
        # move sequences on the right of reference
        translate_y(groups.get_group(i),
                    groups.get_group(i+1),
                    right=True)
    for i in range(reference, 0, -1):
        # move sequences on the left of reference
        translate_y(groups.get_group(i),
                    groups.get_group(i-1),
                    right=False)
    return series


file_path='F:/2023 SA Fynbos Field Work/ASD Spectra of All species/Using Optical Probe/Phylica Dodii/leaf'
list_of_file=os.listdir(file_path)

# Create a new directory for corrected files
output_directory = os.path.join(file_path, 'corrected_asd_files')
os.makedirs(output_directory, exist_ok=True)

# Assuming you have a function called `read_asd_file` to read and process each file
for file_name in list_of_file:
    # Construct the full file path
    file_full_path = os.path.join(file_path, file_name)

    # Call your function to read and process the file
    wavelength, reflectance = read_asd_file(file_full_path)
    # Create a pd.Series with wavelength as the index and reflectance as the values
    series = pd.Series(reflectance, index=wavelength)
    # Define the splices, reference band position, and method
    splices = [1000, 1800]  # List of wavelength values where jumps occur
    reference_band = 0  # Reference band position (0-based)
    correction_method = "additive"  # Method for correction
    # Perform jump correction
    corrected_series = jump_correct(series, splices, reference_band, method=correction_method)
    # Define the output file path in the corrected_asd_files directory
    output_file_path = os.path.join(output_directory, f'{file_name.replace(".asd.txt", "_corrected.txt")}')

    # Save the corrected series and wavelength to a text file
    corrected_series.to_csv(output_file_path, header=False, sep='\t')
    print(f'Saved corrected data to {output_file_path}')
    
    
#%% Ploting corrected ASD files


import os
import pandas as pd
import matplotlib.pyplot as plt

# Directory containing the corrected ASD files
#file_path = 'F:/2023 SA Fynbos Field Work/ASD Spectra of All species/2022 Burn Plot/Background Soil 2022/corrected_asd_files'
file_path = 'F:/2023 SA Fynbos Field Work/ASD Spectra of All species/Using Optical Probe/Erica Irregularis/Dry Flower/corrected_asd_files'

# Get the list of corrected ASD files in the directory
list_of_files = [file for file in os.listdir(file_path) if file.endswith('_corrected.txt')]

# Initialize lists to store reflectance and wavelength data
reflectance_list = []
wavelength_list = []

# Iterate through the list of corrected files
for file_name in list_of_files:
    # Construct the full file path
    file_full_path = os.path.join(file_path, file_name)

    # Read the corrected data from the file into a pd.Series
    corrected_series = pd.read_csv(file_full_path, header=None, delimiter='\t', names=['Wavelength', 'Reflectance'], index_col=0)

    # Append the reflectance and wavelength data to the lists
    reflectance_list.append(corrected_series['Reflectance'])
    wavelength_list.append(corrected_series.index)

# Create a plot
plt.figure()
#https://note.nkmk.me/en/python-for-enumerate-zip/
# Iterate through the reflectance and wavelength data and plot them with labels
for i, (wavelength, reflectance) in enumerate(zip(wavelength_list, reflectance_list), start=1):
    label = f'Data {i}'  # Legend label for each dataset
    #label = file_name.replace('_corrected.txt', '')  # Extract file name without "_corrected.txt" for the legend
    plt.plot(wavelength, reflectance, label=label)

# Add a legend
plt.legend()

# Set axis labels
plt.xlabel('Wavelength')
plt.ylabel('Reflectance')

# Show the plot
plt.show()




