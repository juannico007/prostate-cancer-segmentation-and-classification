import glob
import cv2 as cv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import SimpleITK as sitk
import torchvision.transforms as transforms
# Define a simple custom dataset loader for demonstration purposes
class MRIDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        # Initialize dataset
        self.data_dir = data_dir
        self.transform = transform

        # Load and preprocess data (replace this with your own dataset loading and preprocessing)
        self.data = []  # List to store data samples
        for i in range(len(glob.glob(data_dir+'/3D/*.nii.gz'))):  # Assume 1000 samples
            # Example: Load 3D and 2D images (replace this with your own data loading)
            image_3d_path = f"{data_dir}/3D/{i}.nii.gz"
            label_3d_path = f"{data_dir}/3D/{i}.nii.gz"
            print('loaded 3d images')
            image_2d_path = f"{data_dir}/2D/{i}.jpg"
            print('loaded 2d images')
            label_img = sitk.ReadImage(label_3d_path)

            tumor_middle = label_img.GetSize()[2] // 2
            label_img = sitk.GetArrayFromImage(label_img)
            slice_tumor = label_img[tumor_middle, :, :]
            print('test', label_img.shape, slice_tumor.shape)
            tumor_coordinates = np.where(slice_tumor > 0)
            print('tumor coordinates', tumor_coordinates[1])
            x_min= np.min(tumor_coordinates[0])
            y_min = np.min(tumor_coordinates[1])
            x_max = np.max(tumor_coordinates[0])
            y_max = np.max(tumor_coordinates[1])
            image_3d = sitk.ReadImage(image_3d_path)
            image_3d = sitk.GetArrayFromImage(image_3d)
            #z_slice = image_3d.GetSize()[2] // 2  # Get the index of the middle slice
            image_3d = image_3d[tumor_middle, :, :]
            print('size: ', image_3d.shape)
            #image_3d = image_3d[x_min:x_max, y_min:y_max]
            #region_to_crop = [x_min, y_min]  # Start index for the crop
            image_3d = image_3d[y_min:y_max, x_min:x_max]  # Size of the crop
            image_3d = sitk.GetImageFromArray(image_3d)
            #new_shape = (100, 100)  # Example: 3 rows, 2 columns
            new_size = (200, 200)  # Example: new size (width, height, depth)

            # Perform resampling
            image_3d = sitk.Resample(image_3d, new_size)
            #resized_arr = np.reshape(image_3d, new_shape)
            image_3d = sitk.GetArrayFromImage(image_3d)

            # Crop the image
            #image_3d = sitk.Crop(image_3d, region_to_crop, size_to_crop)

            image_3d = image_3d.astype(np.float32)
            image_3d = torch.from_numpy(image_3d).to(torch.float32)
            print('reading 3d images', image_3d.shape)
            image_2d = cv.imread(image_2d_path)
            image_2d = cv.cvtColor(image_2d, cv.COLOR_BGR2GRAY)
            #image_2d = sitk.ReadImage(image_2d_path)
            #image_2d = sitk.GetArrayFromImage(image_2d)
            image_2d = np.array(image_2d)
            #image_2d = sitk.GetImageFromArray(image_2d)
            #image_2d = sitk.GetArrayFromImage(image_2d)
            image_2d = image_2d[y_min:y_max, x_min:x_max]
            #resized_arr1 = np.reshape(image_2d, new_shape)
            resized_image1 = cv.resize(image_2d,new_size)
            image_2d = resized_image1
            image_2d = torch.from_numpy(image_2d).to(torch.float32)
            print('Reading 2d images', image_2d.shape)

            # Example: Apply transformations (replace this with your own transformations)
            if self.transform:
                image_3d = self.transform(image_3d)
                image_2d = self.transform(image_2d)


            self.data.append((image_3d, image_2d))
            print('data loaded')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Define transformations (replace with your own transformations)

transform = transforms.Compose([
    transforms.ToTensor()
    #transforms.Normalize((0.5,), (0.5,))
])

# Initialize dataset and data loader


