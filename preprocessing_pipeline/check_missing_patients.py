import os

for patient in os.listdir("Data/converted_nii_images"):
    for scan in os.listdir(os.path.join("Data/converted_nii_images", patient)):
        if "study_27" in scan:
            print(patient)