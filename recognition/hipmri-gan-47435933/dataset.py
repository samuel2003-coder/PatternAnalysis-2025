import os
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision.transforms as transforms


class HipMRIDataset(Dataset):
    def __init__(self, root_dir, img_size=64):
        # Folder containing the files
        self.root_dir = root_dir
        # Recursively find all .nii.gz files
        self.img_paths = []
        for root, _, files in os.walk(root_dir):
            for f in files:
                # nii.gz is the filetype of MRIs
                if f.lower().endswith(".nii.gz"):
                    self.img_paths.append(os.path.join(root, f))

        # Define preprocessing
        # Convert image to tensor and then normalise to [-1, 1] range
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.img_size = img_size

    # Return how many images in dataset. Pytorch needs to know how many items exist
    def __len__(self):
        return len(self.img_paths)

    # Loading and preparing image at index idx
    # Reads the MRI image and converts it to numpy array
    def __getitem__(self, idx):
        file_path = self.img_paths[idx]
        img_data = nib.load(file_path).get_fdata().astype(np.float32)

        # Normalize MRI intensities to [0,1]
        img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min() + 1e-8)

        # Resize
        img_data = torch.tensor(img_data).unsqueeze(0)
        img_data = torch.nn.functional.interpolate(
            img_data.unsqueeze(0),
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        # Normalize to [-1, 1]
        img_data = self.transform(img_data)
        # Image is ready to be sent to Discriminator
        return img_data

# Loads the data in batches and shuffles the order
def get_loader(batch_size=32, img_size=64, shuffle=True):
    dataset = HipMRIDataset(img_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    return loader

