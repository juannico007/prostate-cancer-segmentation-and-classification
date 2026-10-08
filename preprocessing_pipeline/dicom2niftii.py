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

input_folder = "Data/raw_images"  
output_folder = "Data/converted_nii_images"
subfolder = "PETMR-WB-PSMA"

def find_pet_folder(patient_folder):
    pet_modalities = [
        "PET-PSMA-PROSTATA",
        "COREG-PET-PSMA-PROSTATA",
        "COREG-E-1-PET-PSMA-PROSTATA-MRJ",
        "E-1-PET-PSMA-PROSTATA",
        "COREG-PET-PSMA-PROSTATA1"
    ]
    
    for modality in pet_modalities:
        pet_folder = os.path.join(patient_folder, subfolder, modality)
        if os.path.exists(pet_folder):
            return pet_folder
    
    return None

pet_modalities = [
    "PET-PSMA-PROSTATA",
    "COREG-PET-PSMA-PROSTATA",
    "COREG-E-1-PET-PSMA-PROSTATA-MRJ",
    "E-1-PET-PSMA-PROSTATA",
    "COREG-PET-PSMA-PROSTATA1"
]

modalities = [
    #ADC MRI
    "ADC-10-6-MM-S",
    "COREG-ADC-10-6-MM-S-1",
    "COREG-ADC-10-6-MM-S",
    #T2 MRI
    "AX-T2-FRFSE",
    "AX-T2-FRFSE-SEN",
    "AX-T2-FRFSE-SUPPLERENDE",
    "COREG-AX-T2-FRFSE-MRJ",
    "COREG-AX-T2-FRFSE",
    "COREG-AX-T2-FRFSE1",
    #SEGMENTATIONS
    "RT-STRUCT-PET-PSMA-PROSTATA",
    "SEGMENTATION-AX-T2-FRFSE-SEN",
    "RT-STRUCT-COREG-E-1-PET-PSMA-PROSTATA-MRJ",
    "RT-STRUCT-COREG-PET-PSMA-PROSTATA1",
    "RT-STRUCT-COREG-PET-PSMA-PROSTATA"
]

dicom_files = os.listdir(input_folder)

for patient in tqdm(dicom_files, desc="Converting DICOM to NIfTI"):
    patient_folder = os.path.join(input_folder, patient)
    pet_folder = find_pet_folder(patient_folder)
    if not pet_folder:
        print(f"Warning: PET folder '{pet_folder}' does not exist for patient '{patient}'. Skipping.")
        print()
        continue
    print(pet_folder)
    print()
    try:
        data = dcmread(os.path.join(pet_folder, sorted(os.listdir(pet_folder))[0]))
        study_id = data['StudyID'].value
        print(study_id)
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
        patient_output_folder = os.path.join(output_folder, study_id)
        os.makedirs(patient_output_folder, exist_ok=True)

        # Convert DICOM to NIfTI
        if modality == "RT-STRUCT-PET-PSMA-PROSTATA" or modality == "RT-STRUCT-COREG-E-1-PET-PSMA-PROSTATA-MRJ"\
            or modality == "RT-STRUCT-COREG-PET-PSMA-PROSTATA1" or modality == "RT-STRUCT-COREG-PET-PSMA-PROSTATA1"\
            or modality == "RT-STRUCT-COREG-PET-PSMA-PROSTATA":
            print(patient, modality)
            #The segmentations are stored in a RTstruct file and they are directly
            #mapped on top ofthe PET-PSMA-PROSTATA scan.
            rtstruct_path = os.path.join(scan_folder, sorted(os.listdir(scan_folder))[0])
            convert_rtstruct(pet_folder, rtstruct_path, f"study_{study_id}_", patient_output_folder)
        else:
            if modality == "SEGMENTATION-AX-T2-FRFSE-SEN":
                modality_short = "Lesion"
            elif "PET-PSMA-PROSTATA" in modality and "RT-STRUCT" not in modality:
                modality_short = "PET"
            elif "AX-T2" in modality:
                modality_short = "MRI_T2"
            elif "ADC-10-6" in modality:
                modality_short = "MRI_ADC"
            command = f"dcm2niix -z y -f study_{study_id}_{modality_short} -o {patient_output_folder} {scan_folder}"
            subprocess.run(command, shell=True, check=True)
    #Create PET 
    command = f"dcm2niix -z y -f study_{study_id}_PET -o {patient_output_folder} {pet_folder}"
    subprocess.run(command, shell=True, check=True)
