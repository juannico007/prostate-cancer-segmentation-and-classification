import os
from matplotlib.pylab import rint
import pydicom
from pydicom import dcmread
from platipy.dicom.io.rtstruct_to_nifti import convert_rtstruct
from tqdm import tqdm
import subprocess


"""Replace by folder with all the patients having 6 series: 
ADC-10-6-MM-S, AX-T2-FRFSE, AX-T2-FRFSE-SEN, PET-PSMA-PROSTATA, 
RT-STRUCT-PER-PSMA-PROSTATA, SEGMENTATION-AX-T2-FRFSE-SEN
Which are the 6 series extracted from hermes"""

input_folder = "Reloads"  
output_folder = "converted_nii_images"
subfolder = "PETMR-WB-PSMA"

modalities = [
    "ADC-10-6-MM-S",
    "AX-T2-FRFSE",
    "AX-T2-FRFSE-SEN",
    "AX-T2-FRFSE-SUPPLERENDE",
    "PET-PSMA-PROSTATA",
    "RT-STRUCT-PET-PSMA-PROSTATA",
    "SEGMENTATION-AX-T2-FRFSE-SEN"
]

dicom_files = os.listdir(input_folder)

for patient in tqdm(dicom_files, desc="Converting DICOM to NIfTI"):
    patient_folder = os.path.join(input_folder, patient)
    pet_folder = os.path.join(patient_folder, subfolder, "PET-PSMA-PROSTATA")
    if not os.path.exists(pet_folder):
        print(f"Warning: PET folder '{pet_folder}' does not exist for patient '{patient}'. Skipping.")
        print()
        continue
    print(pet_folder)
    print()
    try:
        data = dcmread(os.path.join(pet_folder, sorted(os.listdir(pet_folder))[0]))
        study_id = data['StudyID'].value
    except Exception as e:
        print(f"Error reading PET data for patient '{patient}': {e}")
        continue

    for modality in modalities:
        scan_folder = os.path.join(patient_folder, subfolder, modality)
        if not os.path.exists(scan_folder):
            print(f"Warning: Scan folder '{scan_folder}' does not exist for patient '{patient}'. Skipping.")
            print()
            continue

        # Create a folder for the patient if it doesn't exist
        patient_output_folder = os.path.join(output_folder, patient)
        os.makedirs(patient_output_folder, exist_ok=True)

        # Convert DICOM to NIfTI
        if modality == "RT-STRUCT-PET-PSMA-PROSTATA":
            #The segmentations are stored in a RTstruct file and they are directly
            #mapped on top ofthe PET-PSMA-PROSTATA scan. 
            pet_path = os.path.join(patient_folder, subfolder, "PET-PSMA-PROSTATA")
            rtstruct_path = os.path.join(scan_folder, sorted(os.listdir(scan_folder))[0])
            convert_rtstruct(pet_path, rtstruct_path, f"{patient}_{modality}_study_{study_id}_", patient_output_folder)
        else:
            command = f"dcm2niix -z y -f {patient}_{modality}_study_{study_id} -o {patient_output_folder} {scan_folder}"
            subprocess.run(command, shell=True, check=True)
