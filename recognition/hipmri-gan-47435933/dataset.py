"""
Data loader for HipMRI prostate cancer dataset
Handles NIfTI (.nii.gz) medical imaging files
"""
import os
import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import random


class HipMRIDataset(Dataset):
    """
    Dataset class for HipMRI 2D slices from NIfTI files
    """
    def __init__(self, data_dir, img_size=128, augment=False):
        """
        Args:
            data_dir: Directory containing the NIfTI files
            img_size: Target image size (height and width)
            augment: Whether to apply data augmentation
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.augment = augment
        
        # This list will store all valid 2D slices as tuples:
        self.slices = []
        # Allowed file extensions for MRI or image data
        valid_extensions = ('.nii', '.nii.gz', '.png', '.jpg', '.jpeg', '.npy', '.npz')
        
        # Check if provided directory exists
        if os.path.exists(data_dir):
            print(f"Scanning directory: {data_dir}")
            
            # Recursively search for files
            nifti_files = []
            for root, dirs, files in os.walk(data_dir):
                for fname in files:
                    # Only process valid file types
                    if fname.lower().endswith(valid_extensions):
                        file_path = os.path.join(root, fname)
                        
                        # Handle NIfTI files
                        if fname.lower().endswith(('.nii', '.nii.gz')):
                            nifti_files.append(file_path)
                        else:
                            # Regular image files
                            self.slices.append(('image', file_path, 0))
            
            # Load NIfTI files and extract slices
            print(f"Found {len(nifti_files)} NIfTI files, extracting 2D slices...")

            # Iterate through the NIfTI files
            for nifti_path in nifti_files:
                try:
                    # Load NIfTI file
                    nifti_img = nib.load(nifti_path)
                    volume = nifti_img.get_fdata()
                    
                    # Extract 2D slices from the volume
                    # Assume slices are along the last axis (adjust if needed)
                    if volume.ndim == 3:
                        for slice_idx in range(volume.shape[2]):
                            slice_data = volume[:, :, slice_idx]
                            # Only keep slices with meaningful data (not all zeros)
                            if slice_data.max() > 0:
                                self.slices.append(('nifti', nifti_path, slice_idx))
                    elif volume.ndim == 2:
                        # Already 2D
                        # Only keep slices with meaningful data (not all zeros)
                        if volume.max() > 0:
                            self.slices.append(('nifti', nifti_path, 0))

                # Throw exception/errors for loading or directory errors
                except Exception as e:
                    print(f"Error loading {nifti_path}: {e}")
        else:
            print(f"ERROR: Directory does not exist: {data_dir}")
        
        print(f"Found {len(self.slices)} valid 2D slices in {data_dir}")
        
        if len(self.slices) == 0:
            print(f"WARNING: No valid slices found!")
            print(f"Directory contents: {os.listdir(data_dir)[:10] if os.path.exists(data_dir) else 'N/A'}")
    
    def __len__(self):
        """
        Return:
            Total number of slices found
        """
        return len(self.slices)
    
    def __getitem__(self, idx):
        """
        Returns:
            Preprocessed image tensor of shape (1, img_size, img_size) for given index idx
            Parameter:
                idx: index of the image/slice in self.slices
            Returns:
                img_tensor: tensor of shape (1, img_size, img_size)
        """
        # Get file info for slices[idx]
        file_type, file_path, slice_idx = self.slices[idx]
        
        # Load image based on file type
        if file_type == 'nifti':
            # Load NIfTI and extract slice
            nifti_img = nib.load(file_path)
            volume = nifti_img.get_fdata()
            
            # Return total number of slices found
            # If 3D, select the slice; if 2D, use the whole volume
            if volume.ndim == 3:
                img_array = volume[:, :, slice_idx]
            else:
                img_array = volume

            # Convert to float32 and normalize slice to [0, 255]
            img_array = img_array.astype(np.float32)
            if img_array.max() > 0:
                img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())
            img_array = (img_array * 255).astype(np.uint8)
            
            # Convert numpy array to PIL Image
            img = Image.fromarray(img_array).convert('L')
            
        elif file_type == 'image':
            # Handle saved numpy arrays
            if file_path.endswith('.npy'):
                img_array = np.load(file_path)
                 # If already normalized to [0,1], scale to [0,255] 
                 # To ensure image is represented as grayscale
                if img_array.max() <= 1.0:
                    img_array = (img_array * 255).astype(np.uint8)
                img = Image.fromarray(img_array).convert('L')
            # Load .npz archive and pick the first array
            elif file_path.endswith('.npz'):
                data = np.load(file_path)
                img_array = data[data.files[0]]
                 # If already normalized to [0,1], scale to [0,255] 
                 # To ensure image is represented as grayscale
                if img_array.max() <= 1.0:
                    img_array = (img_array * 255).astype(np.uint8)
                img = Image.fromarray(img_array).convert('L')
             # Load standard image files (png, jpg, etc.)
            else:
                img = Image.open(file_path).convert('L')
        
        # Resize image
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        
        # Apply augmentation if enabled (only for training)
        if self.augment:
            img = self._augment_image(img)
        
        # Convert to numpy array and normalize to [0, 1]
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Convert to tensor and normalize to [-1, 1] to match generator tanh output
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)
        img_tensor = (img_tensor - 0.5) / 0.5  # Normalize to [-1, 1]
        
        return img_tensor
    
    def _augment_image(self, img):
        """
        Apply data augmentation to training images. It is done to increase diversity of dataset
        By making the model less sensitive to small changes in the data.
        
        Args:
            img: original image
        Returns:
            Augmented augmented image
        """
        # Random horizontal flip (50% chance)
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Random vertical flip (30% chance - less common for medical images)
        if random.random() > 0.7:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        
        # Random rotation (-15 to 15 degrees) (70% chance)
        if random.random() > 0.3:
            angle = random.uniform(-15, 15)
            img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
        
        # Random brightness adjustment (0.85 to 1.15) (70% chance)
        if random.random() > 0.3:
            enhancer = ImageEnhance.Brightness(img)
            factor = random.uniform(0.85, 1.15)
            img = enhancer.enhance(factor)
        
        # Random contrast adjustment (0.85 to 1.15) (70% chance)
        if random.random() > 0.3:
            enhancer = ImageEnhance.Contrast(img)
            factor = random.uniform(0.85, 1.15)
            img = enhancer.enhance(factor)
        
        # Random sharpness adjustment (0.8 to 1.2) (50% chance)
        if random.random() > 0.5:
            enhancer = ImageEnhance.Sharpness(img)
            factor = random.uniform(0.8, 1.2)
            img = enhancer.enhance(factor)
        
        # Random translation (shift by up to 10% of image size) (50% chance)
        if random.random() > 0.5:
            max_shift = int(0.1 * self.img_size)
            shift_x = random.randint(-max_shift, max_shift)
            shift_y = random.randint(-max_shift, max_shift)
            img = img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, shift_x, 0, 1, shift_y),
                resample=Image.BILINEAR,
                fillcolor=0
            )
        
        return img


def get_loader(data_dir, batch_size=32, img_size=128, shuffle=True, num_workers=4, augment=False):
    """
    Creates a DataLoader for the HipMRI dataset
    
    Parameters:
        data_dir: Directory containing the image files
        batch_size: Batch size for training
        img_size: Target image size
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        augment: Whether to apply data augmentation (use True for training)
    
    Returns:
        DataLoader instance
    """
    # Create the dataset object
    # HipMRIDataset handles loading slices, normalization, and optional augmentation
    dataset = HipMRIDataset(data_dir, img_size=img_size, augment=augment)
    
    # Check for images
    if len(dataset) == 0:
        raise ValueError(f"No valid data found in {data_dir}. Please check the directory path and file formats.")
    
    # Create the instance
    loader = DataLoader(
                 dataset,
                 batch_size=batch_size,       # Number of samples per batch
                 shuffle=shuffle,             # Randomly shuffle the dataset (important for training)
                 num_workers=num_workers,     # Number of parallel workers for faster loading
                 pin_memory=True,             # Copy tensors to GPU pinned memory (faster GPU transfers)
                 drop_last=True               # Drop the last incomplete batch to keep batch size consistent
    )
    
    return loader


# Converting images back too 0,1 range so they may be saved and displayed correctly
def denormalize(tensor):
    """
    Denormalize tensor from [-1, 1] to [0, 1]
    
    Args:
        tensor: Image tensor in range [-1, 1]
    Returns:
        Denormalized tensor in range [0, 1]
    """
    return (tensor + 1) / 2


def save_image_grid(images, save_path, nrow=8, normalize=True):
    """
    Save a grid of images from a batch
    
    Args:
        images: Tensor of images (B, C, H, W)
        save_path: Path to save the image
        nrow: Number of images per row
        normalize: Whether to denormalize from [-1, 1] to [0, 1]
    """
    import math
    from PIL import Image as PILImage
    
    if normalize:
        images = denormalize(images)
    
    # Convert to numpy
    images = images.cpu().detach()
    batch_size = images.shape[0]
    
    # Calculate grid size
    ncol = nrow
    nrow_actual = math.ceil(batch_size / ncol)
    
    # Get image dimensions
    _, _, h, w = images.shape
    
    # Create grid with padding
    padding = 2
    grid_h = nrow_actual * h + (nrow_actual + 1) * padding
    grid_w = ncol * w + (ncol + 1) * padding
    grid = np.ones((grid_h, grid_w), dtype=np.float32)  # White padding
    
    # Fill grid
    for idx in range(batch_size):
        row = idx // ncol
        col = idx % ncol
        
        y = row * h + (row + 1) * padding
        x = col * w + (col + 1) * padding
        
        img = images[idx, 0].numpy()
        grid[y:y+h, x:x+w] = img
    
    # Convert to uint8 and save
    grid = (grid * 255).astype(np.uint8)
    grid_img = PILImage.fromarray(grid, mode='L')
    grid_img.save(save_path)
