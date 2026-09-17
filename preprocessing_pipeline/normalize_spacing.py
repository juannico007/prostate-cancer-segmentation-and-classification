import itk
import nibabel as nib
import os
from pathlib import Path, PurePath
from typing import Tuple
from tqdm import tqdm

def check_image_alignment(pet_image_path, mri_image_path):
    pet = nib.load(pet_image_path)
    mri = nib.load(mri_image_path)

    return nib.aff2axcodes(pet.affine) == nib.aff2axcodes(mri.affine)

def resample_to_size(input_image: itk.Image, target_size: Tuple[float, float, float], interpolator = "linear") -> itk.Image:
    """Take the image resample it onto ``volume``.

    Parameters
    ----------
    input_image : itk.Image
        Image to resample.
    target_size : tuple[float, float, float]
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
    new_size = target_size

    itk.OutputWindow.SetGlobalWarningDisplay(False)
    resample = itk.ResampleImageFilter.New(Input=input_image)
    if interpolator == "linear":
        resample.SetInterpolator(itk.LinearInterpolateImageFunction.New(InputImage=input_image))
    elif interpolator == "spline":
        resample.SetInterpolator(itk.BSplineInterpolateImageFunction.New(InputImage=input_image, order=3))
    
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

def process_patient_nifti(patient_folder, output_folder, target_size = None):

    folder = Path(patient_folder)
    pet_image_path = next(folder.rglob("*PET-PSMA-PROSTATA_study*.nii.gz"), None)
    mri_image_path = next(folder.rglob("*AX-T2-FRFSE*.nii.gz"), None)
    lesion_image_paths = list(folder.rglob("*Lesion*.nii.gz"))
    if not lesion_image_paths:
        return

    mri_image_name = PurePath(mri_image_path).name
    pet_image_name = PurePath(pet_image_path).name
    

    output_mri_path = os.path.join(output_folder, mri_image_name)
    output_pet_path = os.path.join(output_folder, pet_image_name)

    if check_image_alignment(pet_image_path, mri_image_path):
        pet_image = itk.imread(pet_image_path)
        mri_image = itk.imread(mri_image_path)
        

        if target_size is not None:
            mri_image = resample_to_size(mri_image, target_size)
        
        pet_resampled = resample_image_to_mri(pet_image, mri_image)
        os.makedirs(output_folder, exist_ok=True)
        itk.imwrite(mri_image, output_mri_path)
        itk.imwrite(pet_resampled, output_pet_path)

        for lesion_image_path in lesion_image_paths:
            
            lesion_image_name = PurePath(lesion_image_path).name
            output_lesion_path = os.path.join(output_folder, lesion_image_name)
            lesion_image = itk.imread(lesion_image_path)
            lesion_resampled = resample_mask_to_reference(lesion_image, mri_image)
            itk.imwrite(lesion_resampled, output_lesion_path)

        

        

for patient in tqdm(os.listdir("converted_nii_images")):
    patient_folder = os.path.join("converted_nii_images", patient)
    output_folder = os.path.join("resampled_images", patient)
    process_patient_nifti(patient_folder, output_folder, target_size=(128, 128, 128))
