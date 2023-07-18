import torch
from torch.utils.data import Dataset
import numpy as np
import os, glob
import cv2
import tifffile as tiff


class MDITDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        files = glob.glob(os.path.join(data_dir, '*.npy.npz'))
        self.data_filenames = files
        self.transform = transform
        self.image_size = 256
        print('Found {} files in {}.'.format(len(self.data_filenames), data_dir))

    def gaussian_noise(self,image):
        r = np.random.uniform(0, 0.04)
        noise = np.random.normal(0, r, size=(self.image_size,self.image_size, 4))
        image = image + noise
        return image

    def normalize_image(self,image):
        image = image.astype(np.float32)
        min_value = np.min(image)
        max_value = np.max(image)
        image = image - min_value
        max_value = max(max_value-min_value, 1)
        image = image / max_value
        return image

    def load_data(self, filename):
        data_tensor = np.load(filename)
        image = data_tensor['B']/255
        label = data_tensor['L']/255
        image = self.gaussian_noise(image)
        image = self.normalize_image(image)

        image = np.swapaxes(image, 0, 2)
        image = np.swapaxes(image, 1, 2)
        return image,label



    def __len__(self):
        return len(self.data_filenames)


    def __getitem__(self, idx):
        image, label = self.load_data(self.data_filenames[idx])
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        return image, label