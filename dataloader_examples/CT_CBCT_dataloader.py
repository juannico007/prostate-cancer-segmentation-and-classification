import os
import random
import numpy as np
import csv
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import pandas as pd
import torchvision.transforms.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class CTDataset(Dataset):
    def __init__(self, CT_path, transform):
        self.CT_slices = self._collect_slices(CT_path)
        self.transform = transform
        
    def _collect_slices(self, dataset_path):
        slice_paths = []
        for subdir in os.listdir(dataset_path):
            subdir_path = os.path.join(dataset_path, subdir)
            if os.path.isdir(subdir_path):
                for slice_name in os.listdir(subdir_path):
                    slice_path = os.path.join(subdir_path, slice_name)
                    slice_paths.append(slice_path)
        return slice_paths
    
    def __len__(self):
        return len(self.CT_slices)
    
    def __getitem__(self, idx):
        CT_path = self.CT_slices[idx]
        CT_slice = Image.open(CT_path).convert("L")
        if self.transform:
            CT_slice = self.transform(CT_slice)
        return CT_slice

class CBCTtoCTDataset(Dataset):
    def __init__(self, CBCT_path, CT_path, transform):
        self.CBCT_path = CBCT_path
        self.CT_path = CT_path
        self.transform = transform
        self.paired_slice_paths = self._find_paired_slices()

    def _find_paired_slices(self):
        paired_paths = []
        ct_volumes = sorted([
            d for d in os.listdir(self.CT_path)
            if os.path.isdir(os.path.join(self.CT_path, d)) and d.startswith('volume-')
        ])

        for ct_volume_name in ct_volumes:
            volume_num_str = ct_volume_name.split('-')[-1]
            cbct_rec_name = f"REC-{volume_num_str}"

            ct_volume_path = os.path.join(self.CT_path, ct_volume_name)
            cbct_rec_path = os.path.join(self.CBCT_path, cbct_rec_name)

            # List and sort slices within the CT volume for deterministic order
            ct_slices = sorted([
                f for f in os.listdir(ct_volume_path)
                if os.path.isfile(os.path.join(ct_volume_path, f)) # Simple file check is okay
            ])

            for slice_name in ct_slices:
                ct_slice_path = os.path.join(ct_volume_path, slice_name)
                cbct_slice_path = os.path.join(cbct_rec_path, slice_name)

                paired_paths.append((cbct_slice_path, ct_slice_path))
        return paired_paths
        
    def __len__(self):
        return len(self.paired_slice_paths)
    
    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("Index out of range")
        cbct_slice_path, ct_slice_path = self.paired_slice_paths[idx]

        cbct_slice = Image.open(cbct_slice_path).convert('L')
        ct_slice = Image.open(ct_slice_path).convert('L')

        if self.transform:
            cbct_slice = self.transform(cbct_slice)
            ct_slice = self.transform(ct_slice)

        return cbct_slice, ct_slice
    
class PreprocessedCBCTtoCTDataset(Dataset):
    def __init__(self, manifest_path, transform=None):
        self.manifest_path = manifest_path
        self.transform = transform
        self.paired_slice_paths = self._load_manifest()

        if not self.paired_slice_paths:
             print(f"Manifest file '{manifest_path}' loaded 0 pairs. Check the file or its generation process.")
        else:
             print(f"Loaded {len(self.paired_slice_paths)} pairs from {manifest_path}")


    def _load_manifest(self):
        """Loads the paired paths from the CSV manifest file."""
        paired_paths = []
        try:
            with open(self.manifest_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)

                for i, row in enumerate(reader):
                    if len(row) == 2:
                        cbct_path, ct_path = row[0], row[1]
                        if not cbct_path or not ct_path:
                             print.warning(f"Empty path found in manifest row {i+2}. Skipping.")
                             continue
                        paired_paths.append((cbct_path, ct_path))
                    else:
                        print.warning(f"Skipping malformed row {i+2} in manifest: {row}")

        except FileNotFoundError:
            print(f"Manifest file not found: {self.manifest_path}")
        except Exception as e:
            print(f"Error reading manifest file {self.manifest_path}: {e}")

        return paired_paths

    def __len__(self):
        return len(self.paired_slice_paths)

    def __getitem__(self, idx):
        if idx >= len(self):
             raise IndexError("Index out of range")

        # Get paths from the pre-loaded list
        cbct_slice_path, ct_slice_path = self.paired_slice_paths[idx]

        try:
            # Load images
            cbct_slice = Image.open(cbct_slice_path).convert('L')
            ct_slice = Image.open(ct_slice_path).convert('L')

            # Apply transforms (individually, as in your last example)
            if self.transform:
                # Ensure your transform can handle single images if applied like this
                cbct_slice = self.transform(cbct_slice)
                ct_slice = self.transform(ct_slice)

            return cbct_slice, ct_slice

        except FileNotFoundError:
            print.error(f"File not found (path from manifest invalid?): {cbct_slice_path} or {ct_slice_path}")
            return None
        except Exception as e:
            print.error(f"Error loading or processing index {idx} ({cbct_slice_path}, {ct_slice_path}): {e}")
            return None
