"""Body-weight SUV scaling for calibrated, static PET NIfTI volumes."""

import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Mapping

from pydicom.dataset import Dataset
from pydicom.valuerep import DA, DT, TM


def suv_metadata_path(pet_path: str | Path) -> Path:
    path = Path(pet_path)
    if not path.name.endswith(".nii.gz"):
        raise ValueError(f"Expected a .nii.gz PET image: {path}")
    return path.with_name(path.name[:-7] + ".suv.json")


def _positive(metadata: Mapping[str, object], key: str) -> float:
    value = metadata.get(key)
    if not isinstance(value, (int, float, str)) or isinstance(value, bool):
        raise ValueError(f"SUV conversion requires numeric {key}.")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"SUV conversion requires finite, positive {key}.")
    return result


def calculate_suv_factor(metadata: Mapping[str, object]) -> float:
    """Scale already rescaled NIfTI Bq/mL values to SUVbw (g/mL).

    DICOM dose is in Bq and weight in kg. START images require dose decayed
    to the series start; ADMIN images reference the injected activity.
    Uncorrected/dynamic PET and vendor-specific counts are not supported.
    """
    units = metadata.get("Units")
    if units == "GML":
        if metadata.get("SUVType") != "BW":
            raise ValueError("GML PET requires explicit SUVType=BW.")
        return 1.0
    if units != "BQML":
        raise ValueError(f"Unsupported PET Units={units!r}; expected BQML or GML/BW.")
    weight = _positive(metadata, "PatientWeight")
    dose = _positive(metadata, "RadionuclideTotalDose")
    correction = metadata.get("DecayCorrection")
    if correction == "START":
        half_life = _positive(metadata, "RadionuclideHalfLife")
        scan_value = metadata.get("SeriesDateTime")
        injection_value = metadata.get("RadiopharmaceuticalStartDateTime")
        if not isinstance(scan_value, str) or not isinstance(injection_value, str):
            raise ValueError("START PET requires series and injection datetimes.")
        scan = datetime.fromisoformat(scan_value)
        injection = datetime.fromisoformat(injection_value)
        elapsed = (scan - injection).total_seconds()
        if elapsed < 0:
            raise ValueError("PET series starts before radiopharmaceutical injection.")
        dose *= math.exp(-math.log(2.0) * elapsed / half_life)
    elif correction != "ADMIN":
        raise ValueError(f"Unsupported PET DecayCorrection={correction!r}.")
    if not math.isfinite(dose) or dose <= 0:
        raise ValueError("Decay-corrected PET dose must be finite and positive.")
    factor = weight * 1000.0 / dose
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("PET SUV scale factor must be finite and positive.")
    return factor


def metadata_from_dicom(dataset: Dataset) -> dict[str, object]:
    """Extract only quantitative metadata; do not copy patient identifiers."""
    metadata: dict[str, object] = {
        "Units": str(dataset.get("Units", "")),
        "SUVType": str(dataset.get("SUVType", "")),
    }
    if metadata["Units"] == "GML":
        calculate_suv_factor(metadata)
        return metadata
    if metadata["Units"] != "BQML":
        calculate_suv_factor(metadata)
    corrected = dataset.get("CorrectedImage", [])
    if "ATTN" not in corrected or "DECY" not in corrected:
        raise ValueError("SUVbw requires attenuation- and decay-corrected PET.")
    series_type = dataset.get("SeriesType", [])
    first_type = series_type if isinstance(series_type, str) else (
        series_type[0] if series_type else ""
    )
    if first_type not in ("STATIC", "WHOLE BODY"):
        raise ValueError("SUV conversion requires a STATIC or WHOLE BODY PET series.")
    sequence = dataset.get("RadiopharmaceuticalInformationSequence", [])
    if len(sequence) != 1:
        raise ValueError("Expected exactly one radiopharmaceutical for SUV conversion.")
    tracer = sequence[0]
    metadata.update({
        "PatientWeight": str(dataset.get("PatientWeight", "")),
        "RadionuclideTotalDose": str(tracer.get("RadionuclideTotalDose", "")),
        "DecayCorrection": str(dataset.get("DecayCorrection", "")),
    })
    if metadata["DecayCorrection"] == "START":
        metadata["RadionuclideHalfLife"] = str(tracer.get("RadionuclideHalfLife", ""))
        date = DA(dataset.get("SeriesDate"))
        time = TM(dataset.get("SeriesTime"))
        if date is None or time is None:
            raise ValueError("START PET requires SeriesDate and SeriesTime.")
        scan = datetime.combine(date, time)
        injection_dt = tracer.get("RadiopharmaceuticalStartDateTime")
        if injection_dt:
            injection = DT(injection_dt)
            if injection is None:
                raise ValueError("Invalid radiopharmaceutical injection datetime.")
            if injection.tzinfo is not None:
                timezone = dataset.get("TimezoneOffsetFromUTC")
                if not timezone:
                    raise ValueError("Timezone-aware injection requires TimezoneOffsetFromUTC.")
                scan_dt = DT(str(dataset.SeriesDate) + str(dataset.SeriesTime) + str(timezone))
                if scan_dt is None:
                    raise ValueError("Invalid PET series datetime.")
                scan = scan_dt
        else:
            injection_time = TM(tracer.get("RadiopharmaceuticalStartTime"))
            if injection_time is None:
                raise ValueError("START PET requires radiopharmaceutical injection time.")
            # Time-only DICOM injection values refer to the preceding 24 hours.
            injection = datetime.combine(date, injection_time)
            if injection > scan:
                injection -= timedelta(days=1)
        metadata["SeriesDateTime"] = scan.isoformat()
        metadata["RadiopharmaceuticalStartDateTime"] = injection.isoformat()
    calculate_suv_factor(metadata)
    return metadata


def write_suv_metadata(dataset: Dataset, pet_path: str | Path) -> None:
    metadata = metadata_from_dicom(dataset)
    if not Path(pet_path).is_file():
        raise FileNotFoundError(f"Converted PET image not found: {pet_path}")
    with suv_metadata_path(pet_path).open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
        stream.write("\n")


def load_suv_factor(pet_path: str | Path) -> float:
    path = suv_metadata_path(pet_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing PET SUV metadata: {path}. Rerun DICOM conversion to create it."
        )
    with path.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if not isinstance(metadata, dict):
        raise ValueError(f"Expected a PET SUV metadata object in {path}.")
    return calculate_suv_factor(metadata)
