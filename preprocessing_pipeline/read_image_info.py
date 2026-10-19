import pydicom
import os
from collections import defaultdict, Counter

folder = "Data/raw_images/ANON14813304513420240118/PETMR-WB-TUM/PET-PSMA-PROSTATA"

#folder = "Data/raw_images/ANON14813304513420250512/PETMR-WB-PSMA/PET-PSMA-PROSTATA"#AX-T2-FRFSE"
series = defaultdict(list)

for filename in os.listdir(folder):

    path = os.path.join(folder, filename)

    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True)

        uid = ds.get("SeriesInstanceUID")

        series[uid].append({
            "file": filename,
            "series_number": ds.get("SeriesNumber"),
            "description": ds.get("SeriesDescription"),
            "instance": ds.get("InstanceNumber"),
            "position": ds.get("ImagePositionPatient"),
        })

    except Exception:
        pass

print("Number of SeriesInstanceUIDs:", len(series))

for uid, files in series.items():

    print("\nSeries:", uid)
    print("Number of files:", len(files))
    print("Series number:", files[0]["series_number"])
    print("Description:", files[0]["description"])

fields = [
    "SeriesInstanceUID",
    "SeriesNumber",
    "SeriesDescription",
    "ImageType",
    "AcquisitionNumber",
    "InstanceNumber",
    "FrameOfReferenceUID",
    "PixelSpacing",
    "SliceThickness",
    "SpacingBetweenSlices",
    "ImageOrientationPatient",
    "StudyID"
]

values = {field: [] for field in fields}

for filename in os.listdir(folder):
    path = os.path.join(folder, filename)

    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True)

        for field in fields:
            values[field].append(str(ds.get(field)))

    except Exception:
        pass


for field in fields:
    unique = Counter(values[field])

    print(f"\n{'='*50}")
    print(field)
    print(f"Number of different values: {len(unique)}")

    for value, count in unique.items():
        print(f"  {count} files -> {value}")

print("Number of DICOM files:", len(values["InstanceNumber"]))
print("Instance numbers:", sorted([int (i) for i in values["InstanceNumber"]])) 