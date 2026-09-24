"""
This file is in charge of preprocessing the csv file provided by the medical team
It filters the necessary columns from it and formats the column structure of multiple lesions into 
a row structure where each one of the lesions corresponds to one row.

It also adds the bounding box information per each one of the lesions
"""
import itk
import numpy as np
import os
import pandas as pd
from pathlib import Path

def get_bbox(study_id, lesion_id):
    lesion_image_path = Path(os.path.join(f"Data/resampled_images/{study_id}", f"study_{study_id}_Lesion_{lesion_id}.nii.gz"))
    if not lesion_image_path.exists():
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    lesion_image = itk.imread(lesion_image_path)
    mask = itk.GetArrayFromImage(lesion_image)
    coords = np.array(np.where(mask > 0))
    z_min, y_min, x_min = coords.min(axis=1)
    z_max, y_max, x_max = coords.max(axis=1)
    return x_min, y_min, z_min, x_max, y_max, z_max

data = pd.read_csv("Data/PSMAPETMRBiopsyGuida_DATA_LABELS_2026-09-16_2209Nicolas.csv", delimiter = ";")

#Divide the data corresponding to metadata of patients from the data corresponding to the lesions itself
patient_data = data[["Record ID", "Patient alder ved inklusion", "PSA ved skanning", "Prostata volumen", "PSA densitet",
                    "Tidligere MR" ,"Tidligere MRI fund", "Tidligere patologi fra prostata", "Tidligere patologi svar"]]

lesion_1_data = data[["Record ID", 'PRIMARY score lesion 1', "'Red flags' lesion 1", 'SUVmax lesion 1', 'PIRADS Score læsion 1', 
                    'Zone læsion 1 (choice=Peripheral zone)', 'Zone læsion 1 (choice=Transitional zone)', 'Side lesion 1']]
 
lesion_2_data = data[["Record ID", 'PRIMARY score lesion 2', "'Red flags' lesion 2", 'SUVmax lesion 2', 'PIRADS Score læsion 2', 
                    'Zone læsion 2 (choice=Peripheral zone)', 'Zone læsion 2 (choice=Transitional zone)', 'Side lesion 2']]

lesion_3_data = data[["Record ID", 'PRIMARY score lesion 3', "'Red flags' lesion 3", 'SUVmax lesion 3', 'PIRADS Score læsion 3', 
                    'Zone læsion 3 (choice=Peripheral zone)', 'Zone læsion 3 (choice=Transitional zone)', 'Side lesion 3']]

lesion_4_data = data[["Record ID", 'PRIMARY score lesion 4', "'Red flags' lesion 4", 'SUVmax lesion 4', 'PIRADS Score læsion 4', 
                    'Zone læsion 4 (choice=Peripheral zone)', 'Zone læsion 4 (choice=Transitional zone)', 'Side lesion 4']]


#Delete all the nans from the lesions data, to leavo only the actual lesions of the patients
lesion_1_data.dropna(subset= ["PRIMARY score lesion 1"] ,inplace=True)
lesion_2_data.dropna(subset= ["PRIMARY score lesion 2"] ,inplace=True)
lesion_3_data.dropna(subset= ["PRIMARY score lesion 3"] ,inplace=True)
lesion_4_data.dropna(subset= ["PRIMARY score lesion 4"] ,inplace=True)

#Rename the columns in the dataframes for us to be able to marge them as different rows:

lesion_1_data_mapper = {'PRIMARY score lesion 1': 'PRIMARY', 
                        "'Red flags' lesion 1": "Red_flags", 
                        'SUVmax lesion 1': 'SUVmax', 
                        'PIRADS Score læsion 1': 'PIRADS', 
                        'Zone læsion 1 (choice=Peripheral zone)': 'Peripheral', 
                        'Zone læsion 1 (choice=Transitional zone)': 'Transitional', 
                        'Side lesion 1': 'Side'}

lesion_2_data_mapper = {'PRIMARY score lesion 2': 'PRIMARY',
                        "'Red flags' lesion 2": "Red_flags", 
                        'SUVmax lesion 2': 'SUVmax', 
                        'PIRADS Score læsion 2': 'PIRADS', 
                        'Zone læsion 2 (choice=Peripheral zone)': 'Peripheral',
                        'Zone læsion 2 (choice=Transitional zone)': 'Transitional', 
                        'Side lesion 2': 'Side'}

lesion_3_data_mapper = {'PRIMARY score lesion 3': 'PRIMARY',
                        "'Red flags' lesion 3": "Red_flags", 
                        'SUVmax lesion 3': 'SUVmax',
                        'PIRADS Score læsion 3': 'PIRADS',
                        'Zone læsion 3 (choice=Peripheral zone)': 'Peripheral',
                        'Zone læsion 3 (choice=Transitional zone)': 'Transitional',
                        'Side lesion 3': 'Side'}

lesion_4_data_mapper = {'PRIMARY score lesion 4': 'PRIMARY',
                        "'Red flags' lesion 4": "Red_flags", 
                        'SUVmax lesion 4': 'SUVmax',
                        'PIRADS Score læsion 4': 'PIRADS', 
                        'Zone læsion 4 (choice=Peripheral zone)': 'Peripheral',
                        'Zone læsion 4 (choice=Transitional zone)': 'Transitional', 
                        'Side lesion 4': 'Side'}

lesion_1_data.rename(columns = lesion_1_data_mapper, inplace = True)
lesion_2_data.rename(columns = lesion_2_data_mapper, inplace = True)
lesion_3_data.rename(columns = lesion_3_data_mapper, inplace = True)
lesion_4_data.rename(columns = lesion_4_data_mapper, inplace = True)

#Add the number of the lesion to be able to map it
lesion_1_data[["lesion_id"]] = 1
lesion_2_data[["lesion_id"]] = 2
lesion_3_data[["lesion_id"]] = 3
lesion_4_data[["lesion_id"]] = 4

#Concat everything
lesion_data = pd.concat([lesion_1_data, lesion_2_data, lesion_3_data, lesion_4_data])
lesion_data.sort_values(by=["Record ID", "lesion_id"], inplace = True)

#Merge the data
final_data = pd.merge(patient_data, lesion_data, on="Record ID")

#Format names of columns and values
data_mapper = {"Record ID": "study_id",
               "Patient alder ved inklusion": "Age",
               "PSA ved skanning": "PSA_level",
               "Prostata volumen": "Prostata_volume",
               "PSA denistet": "PSA_density",
               "Tidligere MR": "Earlier_MR",
               "Tidligere MRI fund": "Earlier_MRI_finding_PIRADS",
               "Tidligere patologi fra prostata": "Earlier_pathology",
               "Tidligere patologi svar": "Earlier_pathology_answer"}
final_data.rename(columns = data_mapper, inplace = True)


#Here are the transformations made in format that are important to consider for the model
final_data["Earlier_MR"] = final_data["Earlier_MR"].map({"Yes": 1, "No": 0})
final_data["Earlier_pathology"] = final_data["Earlier_pathology"].map({"Yes": 1, "No": 0})
final_data["Earlier_pathology_answer"] = final_data["Earlier_pathology_answer"].map({"Malign": 1, "Benign": 0})
final_data["Red_flags"] = final_data["Red_flags"].map({"Yes": 1, "No": 0})
final_data["Peripheral"] = final_data["Peripheral"].map({"Checked": 1, "Unchecked": 0})
final_data["Transitional"] = final_data["Transitional"].map({"Checked": 1, "Unchecked": 0})
final_data["Side"] = final_data["Side"].map({"højre": 0, "venstre": 1, "midt": 2, "begge": 3})
final_data['Earlier_MRI_finding_PIRADS'] = final_data['Earlier_MRI_finding_PIRADS'].map(lambda x: x.lstrip('PIRADS '))
final_data['PRIMARY'] = final_data['PRIMARY'].map(lambda x: x.lstrip('PRIMARY '))

#Now we need to add the information about the bounding boxes for the 
#network to do their detections

x_min_arr = []
x_max_arr = []
y_min_arr = []
y_max_arr = []
z_min_arr = []
z_max_arr = []

for index, row in final_data.iterrows():
    x_min, y_min, z_min, x_max, y_max, z_max = get_bbox(str(row[0]), row[-1])
    x_min_arr.append(x_min)
    y_min_arr.append(y_min)
    z_min_arr.append(z_min)
    x_max_arr.append(x_max)
    y_max_arr.append(y_max)
    z_max_arr.append(z_max)

final_data["x_min"] = x_min_arr
final_data["y_min"] = y_min_arr
final_data["z_min"] = z_min_arr
final_data["x_max"] = x_max_arr
final_data["y_max"] = y_max_arr
final_data["z_max"] = z_max_arr

final_data.to_csv("Data/data.csv", index = False)