"""Resample registered MRI, PET and masks onto a shared MRI-based grid."""

import argparse
import math
from pathlib import Path
from typing import Literal, Sequence
from tqdm import tqdm

import SimpleITK as sitk


Interpolation = Literal["spline", "linear"]
DEFAULT_MASK_PATTERNS = ("*Lesion*.nii.gz",)
DATA_DIR = Path(__file__).resolve().parents[1] / "Data"


def _validate_image(image: sitk.Image) -> None:
    if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError("Expected a scalar 3D image.")
    if any(size < 1 for size in image.GetSize()):
        raise ValueError("Input image must not be empty.")


def _validate_spacing(spacing: Sequence[float]) -> tuple[float, float, float]:
    if len(spacing) != 3:
        raise ValueError("Spacing must contain three values in (x, y, z) order.")
    x, y, z = (float(value) for value in spacing)
    if any(not math.isfinite(value) or value <= 0 for value in (x, y, z)):
        raise ValueError("Spacing values must be finite and greater than zero.")
    return x, y, z


def resample_to_reference(
    image: sitk.Image,
    reference: sitk.Image,
    *,
    is_mask: bool = False,
    interpolator: Interpolation = "spline",
) -> sitk.Image:
    """Match a reference grid; preserve mask labels or return float32 intensities.

    Inputs must already share physical coordinates (be registered). Points
    outside the source image are filled with zero. Masks always use nearest
    neighbor, regardless of the intensity interpolation setting.
    """
    _validate_image(image)
    _validate_image(reference)
    if interpolator not in ("spline", "linear"):
        raise ValueError(f"Unknown intensity interpolator: {interpolator}")
    interpolation = (
        sitk.sitkNearestNeighbor
        if is_mask
        else sitk.sitkBSpline if interpolator == "spline" else sitk.sitkLinear
    )
    source = image if is_mask else sitk.Cast(image, sitk.sitkFloat32)
    return sitk.Resample(
        source,
        reference,
        sitk.Transform(3, sitk.sitkIdentity),
        interpolation,
        0.0,
        source.GetPixelID(),
    )


def resample_to_spacing(
    image: sitk.Image,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    *,
    is_mask: bool = False,
    interpolator: Interpolation = "spline",
) -> sitk.Image:
    """Resample to (x, y, z) spacing in mm, retaining origin and direction.

    The output covers the first through last input voxel centers. Rounding
    upward can extend coverage by less than one output voxel per axis. This
    chooses a physical spacing, not a fixed voxel count.
    """
    _validate_image(image)
    target_spacing = _validate_spacing(spacing)
    size = [
        math.ceil((count - 1) * old / new) + 1
        for count, old, new in zip(
            image.GetSize(), image.GetSpacing(), target_spacing
        )
    ]
    reference = sitk.Image(size, sitk.sitkUInt8)
    reference.SetSpacing(target_spacing)
    reference.SetOrigin(image.GetOrigin())
    reference.SetDirection(image.GetDirection())
    return resample_to_reference(
        image, reference, is_mask=is_mask, interpolator=interpolator
    )


def process_patient_nifti(
    patient_folder: str | Path,
    patient: str,
    output_folder: str | Path,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    *,
    interpolator: Interpolation = "spline",
    mri_pattern: str = "study_{patient}_MRI_T2.nii.gz",
    pet_pattern: str = "study_{patient}_PET.nii.gz",
    mask_patterns: Sequence[str] = DEFAULT_MASK_PATTERNS,
) -> None:
    """Write one patient's MRI, PET and matched masks without changing inputs.

    Patterns are relative to the patient folder; MRI/PET names may contain
    ``{patient}``. Mask glob patterns are searched recursively. Relative file
    names and mask pixel types are preserved in the output.
    """
    folder = Path(patient_folder).resolve()
    output = Path(output_folder).resolve()
    if output == folder or folder in output.parents or output in folder.parents:
        raise ValueError("Input and output patient folders must be separate.")
    _validate_spacing(spacing)
    mri_path = folder / mri_pattern.format(patient=patient)
    pet_path = folder / pet_pattern.format(patient=patient)
    mask_paths = sorted({
        path
        for pattern in mask_patterns
        for path in folder.rglob(pattern)
        if path.is_file() and path not in (mri_path, pet_path)
    })
    for path in (mri_path, pet_path, *mask_paths):
        if not path.is_relative_to(folder):
            raise ValueError(f"Image path must be inside patient folder: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Missing image for patient {patient}: {path}")

    mri = sitk.ReadImage(str(mri_path))
    pet = sitk.ReadImage(str(pet_path))
    #print(f"Original {patient} MRI spacing (x, y, z) mm: {mri.GetSpacing()}")
    #print(f"Original {patient} PET spacing (x, y, z) mm: {pet.GetSpacing()}")
    mri = resample_to_spacing(mri, spacing, interpolator=interpolator)
    pet = resample_to_reference(pet, mri, interpolator=interpolator)
    output.mkdir(parents=True, exist_ok=True)
    for path, image in ((mri_path, mri), (pet_path, pet)):
        destination = output / path.relative_to(folder)
        destination.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(image, str(destination))
    for path in mask_paths:
        mask = sitk.ReadImage(str(path))
        print(f"Original {patient} mask {path.relative_to(folder)} "
              f"spacing (x, y, z) mm: {mask.GetSpacing()}")
        mask = resample_to_reference(mask, mri, is_mask=True)
        destination = output / path.relative_to(folder)
        destination.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(mask, str(destination))
    #print(f"Resampled {patient}: spacing={mri.GetSpacing()}, "
    #      f"size={mri.GetSize()}, masks={len(mask_paths)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path,
                        default=DATA_DIR / "converted_nii_images",
                        help="Directory containing one subfolder per patient.")
    parser.add_argument("--output-dir", type=Path,
                        default=DATA_DIR / "spacing_resampled_images")
    parser.add_argument("--spacing", type=float, nargs=3, default=(0.2, 0.2, 1),
                        metavar=("X", "Y", "Z"), help="Voxel spacing in mm.")
    parser.add_argument("--interpolator", choices=("spline", "linear"),
                        default="spline", help="MRI/PET interpolation only.")
    parser.add_argument("--mri-pattern", default="study_{patient}_MRI_T2.nii.gz")
    parser.add_argument("--pet-pattern", default="study_{patient}_PET.nii.gz")
    parser.add_argument("--mask-patterns", nargs="+", default=DEFAULT_MASK_PATTERNS,
                        help="Recursive mask globs; quote these in the shell.")
    args = parser.parse_args()
    _validate_spacing(args.spacing)
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if (input_dir == output_dir or input_dir in output_dir.parents
            or output_dir in input_dir.parents):
        raise ValueError("Input and output directories must be separate.")
    patients = sorted(path for path in input_dir.iterdir() if path.is_dir())
    if not patients:
        raise ValueError(f"No patient directories found in {input_dir}")
    for folder in tqdm(patients):
        process_patient_nifti(
            folder, folder.name, output_dir / folder.name, args.spacing,
            interpolator=args.interpolator,
            mri_pattern=args.mri_pattern,
            pet_pattern=args.pet_pattern,
            mask_patterns=args.mask_patterns,
        )


if __name__ == "__main__":
    main()
