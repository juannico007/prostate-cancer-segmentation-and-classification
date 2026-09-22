"""
This file is in charge of preprocessing the csv file provided by the medical team
It filters the necessary columns from it and formats the column structure of multiple lesions into 
a row structure where each one of the lesions corresponds to one row.

Creation date:
22/09/2026

Last modification:
22/09/2026
"""

import pandas as pd

data = pd.read_csv("Data/PSMAPETMRBiopsyGuida_DATA_LABELS_2026-09-16_2209Nicolas.csv", delimiter = ";")
print(data.head())
print(list(data.columns))

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

lesion_1_data_mapper = {'PRIMARY score lesion 1': 'PRIMARY score lesion', 
                        "'Red flags' lesion 1": "'Red flags' lesion", 
                        'SUVmax lesion 1': 'SUVmax lesion', 
                        'PIRADS Score læsion 1': 'PIRADS Score lesion', 
                        'Zone læsion 1 (choice=Peripheral zone)': 'Peripheral zone', 
                        'Zone læsion 1 (choice=Transitional zone)': 'Transitional zone', 
                        'Side lesion 1': 'Side lesion'}

lesion_2_data_mapper = {'PRIMARY score lesion 2': 'PRIMARY score lesion',
                        "'Red flags' lesion 2": "'Red flags' lesion", 
                        'SUVmax lesion 2': 'SUVmax lesion', 
                        'PIRADS Score læsion 2': 'PIRADS Score lesion', 
                        'Zone læsion 2 (choice=Peripheral zone)': 'Peripheral zone',
                        'Zone læsion 2 (choice=Transitional zone)': 'Transitional zone', 
                        'Side lesion 2': 'Side lesion'}

lesion_3_data_mapper = {'PRIMARY score lesion 3': 'PRIMARY score lesion',
                        "'Red flags' lesion 3": "'Red flags' lesion", 
                        'SUVmax lesion 3': 'SUVmax lesion',
                        'PIRADS Score læsion 3': 'PIRADS Score lesion',
                        'Zone læsion 3 (choice=Peripheral zone)': 'Peripheral zone',
                        'Zone læsion 3 (choice=Transitional zone)': 'Transitional zone',
                        'Side lesion 3': 'Side lesion'}

lesion_4_data_mapper = {'PRIMARY score lesion 4': 'PRIMARY score lesion',
                        "'Red flags' lesion 4": "'Red flags' lesion", 
                        'SUVmax lesion 4': 'SUVmax lesion',
                        'PIRADS Score læsion 4': 'PIRADS Score lesion', 
                        'Zone læsion 4 (choice=Peripheral zone)': 'Peripheral zone',
                        'Zone læsion 4 (choice=Transitional zone)': 'Transitional zone', 
                        'Side lesion 4': 'Side lesion'}

lesion_1_data.rename(columns = lesion_1_data_mapper, inplace = True)
lesion_2_data.rename(columns = lesion_2_data_mapper, inplace = True)
lesion_3_data.rename(columns = lesion_3_data_mapper, inplace = True)
lesion_4_data.rename(columns = lesion_4_data_mapper, inplace = True)



lesion_1_data[["number of lesion"]] = 1
lesion_2_data[["number of lesion"]] = 2
lesion_3_data[["number of lesion"]] = 3
lesion_4_data[["number of lesion"]] = 4

lesion_data = pd.concat([lesion_1_data, lesion_2_data, lesion_3_data, lesion_4_data])
lesion_data.sort_values(by=["Record ID", "number of lesion"], inplace = True)

print(lesion_data)

final_data = pd.merge(patient_data, lesion_data, on="Record ID")
print(final_data)
final_data.to_csv("Data/data.csv", index = False)