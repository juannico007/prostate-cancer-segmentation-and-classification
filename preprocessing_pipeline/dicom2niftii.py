import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import tempfile

from openpyxl import Workbook, load_workbook
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from pydicom import dcmread
from platipy.dicom.io.rtstruct_to_nifti import convert_rtstruct
from tqdm import tqdm

if __package__:
    from .pet_suv import suv_metadata_path, write_suv_metadata
else:
    from pet_suv import suv_metadata_path, write_suv_metadata


"""Replace by folder with all the patients having 6 series: 
ADC-10-6-MM-S, AX-T2-FRFSE, AX-T2-FRFSE-SEN, PET-PSMA-PROSTATA, 
RT-STRUCT-PER-PSMA-PROSTATA, SEGMENTATION-AX-T2-FRFSE-SEN
Which are the 6 series extracted from hermes"""

input_folder = "Data/raw_images"  
output_folder = "Data/converted_nii_images"
subfolder = "PETMR-WB-TUM"

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

ERROR_COLUMNS = (
    "Timestamp UTC", "Patient folder", "Study ID", "Modality", "Source",
    "Stage", "Error type", "Error", "Action",
)


def log_error(
    log_path: Path, patient: str, study_id: str, modality: str,
    source: Path, stage: str, error: Exception, action: str,
) -> None:
    """Persist every failure immediately; logging failures stop the batch."""
    details = str(error)
    if isinstance(error, subprocess.CalledProcessError):
        details += f"\nstdout: {error.stdout or ''}\nstderr: {error.stderr or ''}"
    print(f"Error [{stage}] {patient}/{modality}: {details}. {action}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    workbook = load_workbook(log_path) if log_path.exists() else Workbook()
    temporary = None
    try:
        sheet = workbook.active
        if sheet is None:
            raise ValueError(f"Missing error worksheet in {log_path}")
        if not log_path.exists():
            sheet.title = "Conversion errors"
            sheet.append(ERROR_COLUMNS)
        elif tuple(cell.value for cell in sheet[1]) != ERROR_COLUMNS:
            raise ValueError(f"Unexpected error log columns in {log_path}")
        values = (
            datetime.now(timezone.utc).isoformat(), patient, study_id, modality,
            str(source), stage, type(error).__name__, details, action,
        )
        row = sheet.max_row + 1
        for column, value in enumerate(values, start=1):
            cell = sheet.cell(
                row=row, column=column,
                value=ILLEGAL_CHARACTERS_RE.sub("", value)[:32767],
            )
            # Treat error messages and identifiers as text, never Excel formulas.
            cell.data_type = "s"
        with tempfile.NamedTemporaryFile(
            dir=log_path.parent, suffix=".xlsx", delete=False
        ) as stream:
            temporary = Path(stream.name)
        workbook.save(temporary)
        temporary.replace(log_path)
    finally:
        workbook.close()
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _first_file(folder: Path) -> Path:
    files = sorted(path for path in folder.iterdir() if path.is_file())
    if not files:
        raise FileNotFoundError(f"No DICOM files found in {folder}")
    return files[0]


def _convert_nifti(scan_folder: Path, output: Path, filename: str) -> None:
    with tempfile.TemporaryDirectory(dir=output, prefix=".conversion_") as temporary:
        staging = Path(temporary)
        subprocess.run(
            ["dcm2niix", "-z", "y", "-w", "1", "-f", filename,
             "-o", str(staging), str(scan_folder)],
            check=True, capture_output=True, text=True,
        )
        expected = staging / f"{filename}.nii.gz"
        if not expected.is_file():
            raise FileNotFoundError(f"dcm2niix did not produce expected image: {expected.name}")
        for path in sorted(staging.iterdir()):
            path.replace(output / path.name)


def run_conversion(
    source: str | Path = input_folder,
    destination: str | Path = output_folder,
    error_log: str | Path | None = None,
) -> None:
    """Convert scans independently, retaining raw PET when SUV metadata fails."""
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Raw and converted image directories must be separate.")
    log_path = Path(error_log).resolve() if error_log is not None else (
        destination.parent / "dicom_conversion_errors.xlsx"
    )
    patients = sorted(path for path in source.iterdir() if path.is_dir())
    for patient_folder in tqdm(patients, desc="Converting DICOM to NIfTI"):
        patient = patient_folder.name
        found = find_pet_folder(patient_folder)
        if found is None:
            log_error(
                log_path, patient, "", "PET", patient_folder, "PET discovery",
                FileNotFoundError("No supported PET folder found"),
                "Patient skipped; original DICOM retained.",
            )
            continue
        pet_folder = Path(found)
        try:
            data = dcmread(_first_file(pet_folder))
            study_id = str(data["StudyID"].value)
            if not study_id or study_id in (".", "..") or "/" in study_id or "\\" in study_id:
                raise ValueError("StudyID must be a nonempty folder name.")
        except Exception as error:
            log_error(
                log_path, patient, "", pet_folder.name, pet_folder, "PET header",
                error, "Patient skipped; original DICOM retained.",
            )
            continue
        output = destination / study_id
        output.mkdir(parents=True, exist_ok=True)
        for modality in modalities:
            scan_folder = patient_folder / subfolder / modality
            if not scan_folder.is_dir():
                print(f"Warning: Scan folder '{scan_folder}' does not exist. Skipping.")
                continue
            try:
                if modality.startswith("RT-STRUCT-"):
                    with tempfile.TemporaryDirectory(dir=output, prefix=".rtstruct_") as temporary:
                        convert_rtstruct(
                            str(pet_folder), str(_first_file(scan_folder)),
                            f"study_{study_id}_", temporary,
                        )
                        for path in sorted(Path(temporary).iterdir()):
                            path.replace(output / path.name)
                else:
                    modality_short = (
                        "Lesion" if modality == "SEGMENTATION-AX-T2-FRFSE-SEN"
                        else "MRI_T2" if "AX-T2" in modality else "MRI_ADC"
                    )
                    _convert_nifti(scan_folder, output, f"study_{study_id}_{modality_short}")
            except Exception as error:
                log_error(
                    log_path, patient, study_id, modality, scan_folder, "Conversion",
                    error, "Failed scan skipped; original DICOM retained.",
                )

        pet_path = output / f"study_{study_id}_PET.nii.gz"
        # A rerun must never associate new/failed PET data with an old SUV sidecar.
        suv_metadata_path(pet_path).unlink(missing_ok=True)
        try:
            _convert_nifti(pet_folder, output, f"study_{study_id}_PET")
        except Exception as error:
            log_error(
                log_path, patient, study_id, pet_folder.name, pet_folder, "PET conversion",
                error, "Failed PET skipped; original DICOM retained; no SUV sidecar.",
            )
            continue
        try:
            write_suv_metadata(data, pet_path)
        except Exception as error:
            suv_metadata_path(pet_path).unlink(missing_ok=True)
            log_error(
                log_path, patient, study_id, pet_folder.name, pet_folder, "SUV metadata",
                error, "Unnormalized PET NIfTI retained; no SUV sidecar.",
            )


if __name__ == "__main__":
    run_conversion()
