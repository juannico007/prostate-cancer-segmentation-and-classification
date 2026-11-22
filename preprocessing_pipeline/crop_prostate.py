import itk
import numpy as np
import os
import pandas as pd
from pathlib import Path
import subprocess
import sys
import tempfile
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
    itk.Image
        A binary mask of the segmented prostate. The full total_mr task runs
        in a temporary directory; only prostate.nii.gz is retained in the
        output folder. Unrelated existing files are not deleted.
    """
    input_image_path = os.path.join(scan_folder, f"study_{patient_id}_MRI_T2.nii.gz")
    with tempfile.TemporaryDirectory(prefix="total_mr_") as temporary:
        subprocess.run(
            [sys.executable, "-m", "totalsegmentator.bin.TotalSegmentator",
             "-i", input_image_path, "-o", temporary, "--task", "total_mr"],
            check=True,
        )
        segmentation = itk.imread(
            str(Path(temporary) / "prostate.nii.gz")
        ).astype(np.uint8)
        output = Path(patient_output_folder)
        output.mkdir(parents=True, exist_ok=True)
        itk.imwrite(segmentation, str(output / "prostate.nii.gz"))
    return segmentation


def crop_prostate(image, mask, resolution=350, resolution_z=30, *, pad=False):
    """
    Crops the prostate region from the given image using the provided mask.

    Parameters
    ----------
    image : itk.Image
        The input image to be cropped.
    mask : itk.Image
        The binary mask indicating the region of interest (prostate).
    resolution : int, optional
        Crop size in voxels along x and y (default: 350).
    resolution_z : int, optional
        Crop size in slices along z (default: 30).
    pad : bool, optional
        Zero-pad boundary crops to the exact requested size (default: False).

    Notes
    -----
    The crop is centered on the mask bounding box without resampling.
    Image boundaries can reduce the output size. Mask regions larger than
    the requested size are cropped.

    Returns
    -------
    itk.Image
        The cropped image containing only the prostate region.
    """
    # Get the bounding box of the mask
    mask_array = itk.GetArrayFromImage(mask)
    coords = np.argwhere(mask_array)
    if not coords.size:
        raise ValueError("Cannot crop around an empty prostate mask.")
    if any(not isinstance(value, (int, np.integer)) or value <= 0
           for value in (resolution, resolution_z)):
        raise ValueError("Crop dimensions must be positive integers.")
    if (tuple(itk.size(image)) != tuple(itk.size(mask))
            or not np.allclose(image.GetSpacing(), mask.GetSpacing())
            or not np.allclose(image.GetOrigin(), mask.GetOrigin())
            or not np.allclose(itk.array_from_matrix(image.GetDirection()),
                               itk.array_from_matrix(mask.GetDirection()))):
        raise ValueError("Image and prostate mask must share the same grid.")
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    current_res = max_coords - min_coords + 1

    # NumPy arrays use (z, y, x), unlike ITK's (x, y, z).
    target_res = np.array([resolution_z, resolution, resolution])
    padding = (target_res - current_res) // 2
    min_coords -= padding
    max_coords = min_coords + target_res

    if pad:
        lower = np.maximum(-min_coords, 0)
        upper = np.maximum(max_coords - np.array(mask_array.shape), 0)
        image = itk.constant_pad_image_filter(
            image, pad_lower_bound=lower[::-1].astype(int).tolist(),
            pad_upper_bound=upper[::-1].astype(int).tolist(), constant=0,
        )
        region = itk.ImageRegion[3]()
        region.SetIndex(min_coords[::-1].astype(int).tolist())
        region.SetSize(target_res[::-1].astype(int).tolist())
        return itk.region_of_interest_image_filter(image, region_of_interest=region)

    # Clamp coordinates to image boundaries
    shape = np.array(mask_array.shape)
    min_coords = np.maximum(min_coords, 0)
    max_coords = np.minimum(max_coords, shape)

    # Get start and size for the region of interest in ITK's coordinate system 
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


def main():
    nii_images = "Data/converted_nii_images"
    for patient in tqdm(os.listdir(nii_images)):
        patient_folder = os.path.join(nii_images, patient)
        output_folder = os.path.join("Data/prostate_masks", patient)
        os.makedirs(output_folder, exist_ok=True)
        mri_image_path = os.path.join(patient_folder, f"study_{patient}_MRI_T2.nii.gz")
        mri_image = itk.imread(mri_image_path)
        prostate_mask = segment_prostate(patient, patient_folder, output_folder)
        cropped_mri_image = crop_prostate(mri_image, prostate_mask)
        output_mri_path = os.path.join(
            patient_folder, f"study_{patient}_new_cropped_MRI_T2.nii.gz"
        )
        itk.imwrite(cropped_mri_image, output_mri_path)


if __name__ == "__main__":
    main()