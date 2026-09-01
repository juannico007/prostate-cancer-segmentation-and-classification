import os
from tqdm import tqdm
import subprocess

"""Replace by folder with all the patients having 4 series: 
ADC-10-6-MM-S, AX-T2-FRFSE, PET-PSMA-PROSTATA, RT-STRUCT-PER-PSMA-PROSTATA
Which are the 4 series extracted"""

input_folder = "testing_images"  
output_folder = "converted_nii_images"

modalities = [
    "ADC-10-6-MM-S",
    "AX-T2-FRFSE",
    "PET-PSMA-PROSTATA",
    "RT-STRUCT-PER-PSMA-PROSTATA"
]

dicom_files = os.listdir(input_folder)

for patient in tqdm(dicom_files, desc="Converting DICOM to NIfTI"):
    for modality in modalities:
        scan_folder = os.path.join(input_folder, patient, "PETMR-WB-PSMA", modality)
        if not os.path.exists(scan_folder):
            print(f"Warning: Scan folder '{scan_folder}' does not exist for patient '{patient}'. Skipping.")
            continue

        # Create a folder for the patient if it doesn't exist
        patient_output_folder = os.path.join(output_folder, patient)
        os.makedirs(patient_output_folder, exist_ok=True)

        # Convert DICOM to NIfTI using dcm2niix
        command = f"dcm2niix -z y -f {patient}_{modality} -o {patient_output_folder} {scan_folder}"
        subprocess.run(command, shell=True, check=True)
