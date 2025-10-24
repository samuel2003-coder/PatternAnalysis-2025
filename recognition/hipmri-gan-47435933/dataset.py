import os
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision.transforms as transforms


class HipMRIDataset(Dataset):
    def __init__(self, root_dir, img_size=64, transform=None):
        self.root_dir = root_dir
        self.img_paths = []
        for root, _, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith(".nii.gz"):
                    self.img_paths.append(os.path.join(root, f))

        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        file_path = self.img_paths[idx]
        img_data = nib.load(file_path).get_fdata().astype(np.float32)

        # Normalize MRI intensities to [0,1]
        img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min() + 1e-8)

        # Convert to tensor with channel dim
        img_tensor = torch.from_numpy(img_data).unsqueeze(0)  # (1, H, W)

        # Apply transform if provided
        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor


def get_loader(root_dir, batch_size=32, img_size=64, shuffle=True):
    # Define preprocessing transform
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.Normalize([0.5], [0.5])  # maps [0,1] → [-1,1]
    ])

    dataset = HipMRIDataset(root_dir, img_size, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    return loader

