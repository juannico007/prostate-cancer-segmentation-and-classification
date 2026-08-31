import torch
import glob
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import SimpleITK as sitk
import os
import torchvision.transforms as transforms

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # Load paths to CT and Dose files
        self.ct_paths = glob.glob(os.path.join(data_dir, 'CTs/re_sized/', '*.nii.gz'))
        self.dose_paths = glob.glob(os.path.join(data_dir, 'Doses/re_sized/', '*.nii.gz'))
        self.struct_paths = glob.glob(os.path.join(data_dir, 'combined_structs/re_sized/', '*.nii.gz'))


    def __len__(self):
        return len(self.ct_paths)

    def __getitem__(self, idx):
        ct_path = self.ct_paths[idx]
        dose_path = self.dose_paths[idx]
        struct_path = self.struct_paths[idx]

        # Read CT and Dose images
        ct_image = sitk.ReadImage(ct_path)
        dose_image = sitk.ReadImage(dose_path)
        struct_img = sitk.ReadImage(struct_path)

        # Convert images to arrays
        ct_array = sitk.GetArrayFromImage(ct_image).astype(np.float32)
        dose_array = sitk.GetArrayFromImage(dose_image).astype(np.float32)
        struct_array = sitk.GetArrayFromImage(struct_img).astype(np.float32)
        ct_array = torch.from_numpy(ct_array).to(torch.float32)
        dose_array = torch.from_numpy(dose_array).to(torch.float32)
        struct_array = torch.from_numpy(struct_array).to(torch.float32)

        # Apply transformations
        if self.transform:
            ct_array = self.transform(ct_array)
            dose_array = self.transform(dose_array)
            struct_array = self.transform(struct_array)

        # Add batch and channel dimensions
        ct_array = ct_array.unsqueeze(0)  # Add batch dimension
        dose_array = dose_array.unsqueeze(0)  # Add batch dimension
        struct_array = struct_array.unsqueeze(0)
        #print('dataloaded: ')

        return ct_array, dose_array, struct_array
