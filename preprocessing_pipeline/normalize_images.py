import itk
import nibabel as nib
import numpy as np
import os
from pathlib import Path, PurePath
from typing import Literal, Tuple
from tqdm import tqdm

def check_image_alignment(pet_image_path, mri_image_path):
    pet = nib.load(pet_image_path)
    mri = nib.load(mri_image_path)

    return nib.aff2axcodes(pet.affine) == nib.aff2axcodes(mri.affine)

def normalize_intensity(image: itk.Image, reference_image: itk.Image, method: Literal["zscore", "minmax", "none"] = "zscore") -> itk.Image:
    """Return a normalised copy of ``image`` using the requested strategy.

    Parameters
    ----------
    image : itk.Image
        Input image to normalise. Spatial metadata is preserved in the output.
    reference_image : itk.Image
        Reference image used for normalization, used for the cropped MRI.
    method : {"zscore", "minmax", "none"}, default="none"
        Name of the intensity scaling strategy to apply.

    Returns
    -------
    itk.Image
        The normalised image (or the original image when ``method`` is ``none``).
    """
    if method == "none":
        return image

    arr = itk.GetArrayFromImage(image).astype(np.float32)
    ref_arr = itk.GetArrayFromImage(reference_image).astype(np.float32)

    if method == "zscore":
        m, s = float(ref_arr.mean()), float(ref_arr.std())
        arr = (arr - m) / s if s > 0 else (arr - m)
    elif method == "minmax":
        lo, hi = float(ref_arr.min()), float(ref_arr.max())
        arr = (arr - lo) / (hi - lo) if hi > lo else (arr - lo)
    else:
        raise ValueError(f"Unknown normalization: {method}")

    out = itk.GetImageFromArray(arr)
    out.CopyInformation(image)
    return out

def resample_to_size(input_image: itk.Image, target_size: Tuple[int, int, int], interpolator = "linear") -> itk.Image:
    """Take the image resample it onto ``volume``.

    Parameters
    ----------
    input_image : itk.Image
        Image to resample.
    target_size : tuple[int, int, int]
        Desired output size (x,y,z)
    interpolator:
        Linear or spline interpolation (will use linear by default as it is more stable)
    Returns
    -------
    itk.Image
        Resampled image with origin and direction preserved from the input.
    """
    input_size = itk.size(input_image)            # e.g. [x, y, z]
    input_spacing = input_image.GetSpacing()      # e.g. (sx, sy, sz) 
    orig_org = input_image.GetOrigin()
    orig_dir = input_image.GetDirection()
    if len(target_size) != 3 or any(
        not isinstance(value, (int, np.integer)) or value <= 0
        for value in target_size
    ):
        raise ValueError("Target size must contain three positive integers.")
    new_size = target_size

    resample = itk.ResampleImageFilter.New(Input=input_image)
    if interpolator == "linear":
        resample.SetInterpolator(itk.LinearInterpolateImageFunction.New(InputImage=input_image))
    elif interpolator == "spline":
        spline = itk.BSplineInterpolateImageFunction.New(InputImage=input_image)
        spline.SetSplineOrder(3)
        resample.SetInterpolator(spline)
    else:
        raise ValueError(f"Unknown interpolator: {interpolator}")
    
    output_spacing = [
    input_spacing[i] * input_size[i] / new_size[i]
    for i in range(3)
    ]
    resample.SetOutputSpacing(output_spacing)
    resample.SetSize(new_size)
    resample.SetOutputOrigin(orig_org)
    resample.SetOutputDirection(orig_dir)
    resample.SetTransform(itk.IdentityTransform[itk.D, 3].New())
    resample.Update()
    return resample.GetOutput()

def resample_image_to_mri(image, mri_image):
    resample = itk.ResampleImageFilter.New(Input=image)

    resample.SetInterpolator(
        itk.LinearInterpolateImageFunction.New(
            InputImage=image
        )
    )

    resample.SetOutputSpacing(mri_image.GetSpacing())
    resample.SetSize(
        itk.size(mri_image)
    )
    resample.SetOutputOrigin(mri_image.GetOrigin())
    resample.SetOutputDirection(mri_image.GetDirection())

    resample.SetTransform(
        itk.IdentityTransform[itk.D, 3].New()
    )

    resample.SetDefaultPixelValue(0.0)

    resample.Update()

    return resample.GetOutput()

def resample_mask_to_reference(
    mask_image: itk.Image,
    reference_image: itk.Image,
) -> itk.Image:

    resample = itk.ResampleImageFilter.New(
        Input=mask_image
    )

    # Nearest-neighbor interpolation for binary labels
    resample.SetInterpolator(
        itk.NearestNeighborInterpolateImageFunction.New(
            InputImage=mask_image
        )
    )

    # Use the MRI as the reference grid
    resample.SetOutputSpacing(reference_image.GetSpacing())
    resample.SetSize(
        reference_image.GetLargestPossibleRegion().GetSize()
    )
    resample.SetOutputOrigin(reference_image.GetOrigin())
    resample.SetOutputDirection(reference_image.GetDirection())

    # Outside the mask
    resample.SetDefaultPixelValue(0)

    # Identity if mask and MRI are already registered
    resample.SetTransform(
        itk.IdentityTransform[itk.D, 3].New()
    )

    resample.Update()

    return resample.GetOutput()

def process_patient_nifti(patient_folder, patient, output_folder, target_size = None):

    folder = Path(patient_folder)
    #This regular expression includes the lesions as well so we need to filter them out afterwards.
    pet_image_path = os.path.join(patient_folder, f"study_{patient}_PET.nii.gz")
    
    cropped_mri_image_path = os.path.join(patient_folder, f"study_{patient}_new_cropped_MRI_T2.nii.gz")
    mri_image_path = os.path.join(patient_folder, f"study_{patient}_MRI_T2.nii.gz")
    lesion_image_paths = list(folder.rglob("*Lesion*.nii.gz"))
    print(lesion_image_paths)
    mri_image_name = PurePath(cropped_mri_image_path).name
    pet_image_name = PurePath(pet_image_path).name
    

    output_mri_path = os.path.join(output_folder, mri_image_name)
    output_pet_path = os.path.join(output_folder, pet_image_name)

    if check_image_alignment(pet_image_path, cropped_mri_image_path):
        pet_image = itk.imread(pet_image_path)
        cropped_mri_image = itk.imread(cropped_mri_image_path)
        mri_image = itk.imread(mri_image_path)
        

        if target_size is not None:
            cropped_mri_image = resample_to_size(cropped_mri_image, target_size)
        
        pet_resampled = resample_image_to_mri(pet_image, cropped_mri_image)

        normalized_mri_image = normalize_intensity(cropped_mri_image, mri_image, method="zscore")
        normalized_pet_image = normalize_intensity(pet_resampled, pet_image, method="zscore")
        
        os.makedirs(output_folder, exist_ok=True)
        itk.imwrite(normalized_mri_image, output_mri_path)
        itk.imwrite(normalized_pet_image, output_pet_path)


        if not lesion_image_paths:
            return
        print("Found lesions for patient: ", patient_folder)
        for lesion_image_path in lesion_image_paths:
            
            lesion_image_name = PurePath(lesion_image_path).name
            output_lesion_path = os.path.join(output_folder, lesion_image_name)
            lesion_image = itk.imread(lesion_image_path)
            lesion_resampled = resample_mask_to_reference(lesion_image, normalized_mri_image)
            itk.imwrite(lesion_resampled, output_lesion_path)
    
def main():
    for patient in tqdm(os.listdir("Data/converted_nii_images")):
        patient_folder = os.path.join("Data/converted_nii_images", patient)
        output_folder = os.path.join("Data/cropped_resampled_images_new", patient)
        print("processing patient: ", patient)
        process_patient_nifti(patient_folder, patient, output_folder, target_size=(128, 128, 128))
        print()


if __name__ == "__main__":
    main()
