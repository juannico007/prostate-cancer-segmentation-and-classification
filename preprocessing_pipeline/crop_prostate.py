import itk
import numpy as np
import os
import pandas as pd
from pathlib import Path
import subprocess
from tqdm import tqdm


def segment_prostate(patient_id, scan_folder, patient_output_folder):
    """
    Segments the prostate from the given image using a TotalSegmentator model. 

    Parameters
    ----------
    patient_id : str
        The ID of the patient.
    scan_folder : str
        The path to the folder containing the scan files.
    patient_output_folder : str
        The path to the folder where the output files will be saved.

    Returns
    -------
    np.ndarray
        A binary mask of the segmented prostate.
    """
    input_image_path = next(Path(patient_folder).rglob("*MRI_T2.nii.gz"), None)
    command = f"TotalSegmentator -i {input_image_path} -o {patient_output_folder} --task total_mr"
    subprocess.run(command, shell=True, check=True)

    # Load the segmentation result
    segmentation_path = os.path.join(patient_output_folder, f"prostate.nii.gz")
    segmentation = itk.imread(segmentation_path)

    #Delete all of the other segmentations except for the prostate
    for file in os.listdir(patient_output_folder):
        if file != "prostate.nii.gz":
            os.remove(os.path.join(patient_output_folder, file))

    return segmentation.astype(np.uint8)


def crop_prostate(image, mask, resolution = 256):
    """
    Crops the prostate region from the given image using the provided mask.

    Parameters
    ----------
    image : itk.Image
        The input image to be cropped.
    mask : itk.Image
        The binary mask indicating the region of interest (prostate).

    Returns
    -------
    itk.Image
        The cropped image containing only the prostate region.
    """
    # Get the bounding box of the mask
    mask_array = itk.GetArrayFromImage(mask)
    coords = np.argwhere(mask_array)
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    current_res = max_coords - min_coords

    padding = (resolution - current_res) // 2
    min_coords -= padding
    max_coords += padding+1

    # Clamp coordinates to image boundaries
    shape = np.array(mask_array.shape)
    min_coords = np.maximum(min_coords, 0)
    max_coords = np.minimum(max_coords, shape)

    # Convert to ITK order (x, y, z)
    start = min_coords[::-1].astype(int).tolist()
    size = (max_coords - min_coords)[::-1].astype(int).tolist()

    # Define the region of interest
    region = itk.ImageRegion[3]()
    region.SetIndex(start)
    region.SetSize(size)

    # Crop the image
    roi_filter = itk.RegionOfInterestImageFilter.New(Input=image)
    roi_filter.SetRegionOfInterest(region)
    roi_filter.Update()

    return roi_filter.GetOutput()


nii_images = "Data/converted_nii_images"
for patient in tqdm(os.listdir(nii_images)):
    patient_folder = os.path.join(nii_images, patient)
    output_folder = os.path.join("Data/prostate_masks", patient)
    os.makedirs(output_folder, exist_ok=True)

    # Load the MRI image
    mri_image_path = next(Path(patient_folder).rglob("*MRI_T2.nii.gz"), None)
    if mri_image_path is None:
        print(f"No MRI image found for patient {patient}. Skipping.")
        continue
    mri_image = itk.imread(mri_image_path)

    # Segment the prostate
    prostate_mask = segment_prostate(patient, patient_folder, output_folder)

    # Crop the prostate region from the MRI image
    cropped_mri_image = crop_prostate(mri_image, prostate_mask)

    # Save the cropped MRI image
    output_mri_path = os.path.join(patient_folder, f"study_{patient}_cropped_MRI_T2.nii.gz")
    itk.imwrite(cropped_mri_image, output_mri_path)