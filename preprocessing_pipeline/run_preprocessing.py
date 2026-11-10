"""Run spacing resampling, prostate cropping, and network input preparation."""

import argparse
from pathlib import Path
import tempfile
from typing import Literal, Sequence

import itk
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

if __package__:
    from . import crop_prostate, normalize_images, resample_images
else:
    import crop_prostate
    import normalize_images
    import resample_images


DATA_DIR = Path(__file__).resolve().parents[1] / "Data"


def _to_itk(image: sitk.Image) -> itk.Image:
    """Bridge the spacing module's SimpleITK images to the existing ITK helpers."""
    result = itk.GetImageFromArray(sitk.GetArrayFromImage(image))
    result.SetSpacing(image.GetSpacing())
    result.SetOrigin(image.GetOrigin())
    result.SetDirection(
        itk.matrix_from_array(np.array(image.GetDirection()).reshape(3, 3))
    )
    return result


def _write(image: itk.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    itk.imwrite(image, str(path))


def _validate_size(size: Sequence[int], name: str) -> None:
    if len(size) != 3 or any(
        not isinstance(value, (int, np.integer)) or value <= 0 for value in size
    ):
        raise ValueError(f"{name} must contain three positive integers in x/y/z order.")


def _check_separate(first: Path, second: Path) -> None:
    if first == second or first in second.parents or second in first.parents:
        raise ValueError(f"Directories must be separate: {first} and {second}")


def preprocess_patient(
    patient_folder: Path,
    output_folder: Path,
    *,
    spacing: Sequence[float] = (0.2, 0.2, 1.0),
    crop_size: Sequence[int] = (350, 350, 30),
    target_size: Sequence[int] = (128, 128, 128),
    interpolator: resample_images.Interpolation = "spline",
    normalization: Literal["zscore", "minmax", "none"] = "zscore",
    mask_patterns: Sequence[str] = resample_images.DEFAULT_MASK_PATTERNS,
    save_intermediate: bool = False,
    intermediate_folder: Path | None = None,
) -> None:
    """Process one patient; only final images persist unless debugging is enabled.

    TotalSegmentator always generates a new prostate mask from the
    spacing-resampled MRI. Temporary segmentation files are removed even on
    failure. Normalization uses full spacing-resampled volumes as references,
    matching the existing helper's full-volume rather than cropped statistics.
    """
    folder = Path(patient_folder).resolve()
    output = Path(output_folder).resolve()
    _check_separate(folder, output)
    resample_images._validate_spacing(spacing)
    _validate_size(crop_size, "Crop size")
    _validate_size(target_size, "Target size")
    if crop_size[0] != crop_size[1]:
        raise ValueError("The prostate crop requires equal x and y sizes.")
    if normalization not in ("zscore", "minmax", "none"):
        raise ValueError(f"Unknown normalization: {normalization}")
    if interpolator not in ("spline", "linear"):
        raise ValueError(f"Unknown interpolator: {interpolator}")
    debug = None
    if save_intermediate:
        debug = (Path(intermediate_folder).resolve() if intermediate_folder is not None
                 else output.parent.with_name(output.parent.name + "_intermediate")
                 / output.name)
        _check_separate(folder, debug)
        _check_separate(output, debug)

    patient = folder.name
    mri_path = folder / f"study_{patient}_MRI_T2.nii.gz"
    pet_path = folder / f"study_{patient}_PET.nii.gz"
    for path in (mri_path, pet_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing image for patient {patient}: {path}")
    masks = sorted({
        path for pattern in mask_patterns for path in folder.rglob(pattern)
        if path.is_file() and path not in (mri_path, pet_path)
    })
    for path in masks:
        if not path.resolve().is_relative_to(folder):
            raise ValueError(f"Mask must be inside patient folder: {path}")

    mri_source = sitk.ReadImage(str(mri_path))
    pet_source = sitk.ReadImage(str(pet_path))
    print(f"Original {patient} MRI spacing (x, y, z) mm: {mri_source.GetSpacing()}")
    print(f"Original {patient} PET spacing (x, y, z) mm: {pet_source.GetSpacing()}")
    grid = resample_images.resample_to_spacing(
        mri_source, spacing, interpolator=interpolator
    )
    mri = _to_itk(grid)
    pet = _to_itk(resample_images.resample_to_reference(
        pet_source, grid, interpolator=interpolator
    ))
    del mri_source, pet_source
    if debug is not None:
        _write(mri, debug / "01_spacing" / mri_path.name)
        _write(pet, debug / "01_spacing" / pet_path.name)

    with tempfile.TemporaryDirectory(prefix=f"prostate_{patient}_") as temporary:
        work = Path(temporary)
        _write(mri, work / mri_path.name)
        prostate = crop_prostate.segment_prostate(patient, work, work / "segmentation")
    prostate = normalize_images.resample_mask_to_reference(prostate, mri)
    if debug is not None:
        _write(prostate, debug / "01_spacing" / "prostate.nii.gz")

    def crop(image: itk.Image) -> itk.Image:
        return crop_prostate.crop_prostate(
            image, prostate, resolution=crop_size[0], resolution_z=crop_size[2],
            pad=True,
        )

    cropped_mri = crop(mri)
    cropped_pet = crop(pet)
    if debug is not None:
        _write(cropped_mri, debug / "02_cropped" / mri_path.name)
        _write(cropped_pet, debug / "02_cropped" / pet_path.name)
        _write(crop(prostate), debug / "02_cropped" / "prostate.nii.gz")

    final_mri = normalize_images.resample_to_size(cropped_mri, tuple(target_size))
    final_pet = normalize_images.resample_image_to_mri(cropped_pet, final_mri)
    normalized_mri = normalize_images.normalize_intensity(
        final_mri, mri, method=normalization
    )
    normalized_pet = normalize_images.normalize_intensity(
        final_pet, pet, method=normalization
    )
    _write(normalized_mri, output / mri_path.name)
    _write(normalized_pet, output / pet_path.name)
    _write(
        normalize_images.resample_mask_to_reference(crop(prostate), final_mri),
        output / "prostate.nii.gz",
    )
    del mri, pet, cropped_mri, cropped_pet, final_pet, normalized_mri, normalized_pet
    # Process lesion masks individually rather than retaining all volumes.
    for path in masks:
        relative = path.relative_to(folder)
        if relative == Path("prostate.nii.gz"):
            continue
        source = sitk.ReadImage(str(path))
        print(f"Original {patient} mask {relative} "
              f"spacing (x, y, z) mm: {source.GetSpacing()}")
        mask = _to_itk(resample_images.resample_to_reference(
            source, grid, is_mask=True
        ))
        if debug is not None:
            _write(mask, debug / "01_spacing" / relative)
        cropped_mask = crop(mask)
        if debug is not None:
            _write(cropped_mask, debug / "02_cropped" / relative)
        _write(
            normalize_images.resample_mask_to_reference(cropped_mask, final_mri),
            output / relative,
        )
    print(f"Preprocessed {patient}: size={tuple(itk.size(final_mri))}, "
          f"spacing={tuple(final_mri.GetSpacing())}")


def run_preprocessing(
    input_dir: str | Path = DATA_DIR / "converted_nii_images",
    output_dir: str | Path = DATA_DIR / "preprocessed_images",
    *,
    spacing: Sequence[float] = (0.2, 0.2, 1.0),
    crop_size: Sequence[int] = (350, 350, 30),
    target_size: Sequence[int] = (128, 128, 128),
    interpolator: resample_images.Interpolation = "spline",
    normalization: Literal["zscore", "minmax", "none"] = "zscore",
    mask_patterns: Sequence[str] = resample_images.DEFAULT_MASK_PATTERNS,
    save_intermediate: bool = False,
) -> None:
    """Run all patient subfolders sequentially, saving only final outputs by default."""
    source = Path(input_dir).resolve()
    output = Path(output_dir).resolve()
    _check_separate(source, output)
    debug = output.with_name(output.name + "_intermediate")
    if save_intermediate:
        _check_separate(source, debug)
        _check_separate(output, debug)
    patients = sorted(path for path in source.iterdir() if path.is_dir())
    if not patients:
        raise ValueError(f"No patient directories found in {source}")
    for folder in tqdm(patients, desc="Preprocessing patients"):
        preprocess_patient(
            folder, output / folder.name, spacing=spacing, crop_size=crop_size,
            target_size=target_size, interpolator=interpolator,
            normalization=normalization, mask_patterns=mask_patterns,
            save_intermediate=save_intermediate,
            intermediate_folder=debug / folder.name,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DATA_DIR / "converted_nii_images")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "preprocessed_images")
    parser.add_argument("--spacing", type=float, nargs=3, default=(0.2, 0.2, 1.0))
    parser.add_argument("--crop-size", type=int, nargs=3, default=(400, 400, 80)) #Default voxels selected after seeing the size of the resampled prostates
    parser.add_argument("--target-size", type=int, nargs=3, default=(128, 128, 128))
    parser.add_argument("--interpolator", choices=("spline", "linear"), default="spline",
                        help="Initial spacing interpolation; network resizing is linear.")
    parser.add_argument("--normalization", choices=("zscore", "minmax", "none"),
                        default="zscore")
    parser.add_argument("--mask-patterns", nargs="+",
                        default=resample_images.DEFAULT_MASK_PATTERNS)
    parser.add_argument("--save-intermediate", action="store_true",
                        help="Persist spacing/crop outputs in <output-dir>_intermediate.")
    args = parser.parse_args()
    run_preprocessing(**vars(args))


if __name__ == "__main__":
    main()
