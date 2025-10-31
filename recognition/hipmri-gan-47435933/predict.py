"""
Prediction and Visualization Script for Trained WGAN-GP Model

This script loads your trained generator and creates comprehensive visualizations:
- Random generated samples
- Latent space interpolations
- Comparisons with real MRI images
- SSIM quality metrics
- Statistical analysis

Usage: python predict.py
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import math

# Import our custom modules
from modules import Generator, compute_ssim
from dataset import denormalize, get_loader


# ============================================================
# CONFIGURATION
# ============================================================

# Model parameters (must match training configuration)
latent_dim = 128        # Size of the random noise vector
img_size = 128          # Output image size (128x128 pixels)
img_channels = 1        # Grayscale MRI images

# Set device (use GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("WGAN-GP TRAINED MODEL - PREDICTION AND VISUALIZATION")
print("=" * 60)
print(f"Device: {device}")


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("\nLoading trained model...")

# Initialize the generator with the same architecture used in training
generator = Generator(latent_dim=latent_dim, img_channels=img_channels).to(device)

# Try to load the best model (highest SSIM), fallback to final model if not found
model_path = 'checkpoints/best_model.pth'
if not os.path.exists(model_path):
    model_path = 'checkpoints/final_model.pth'
    print(f"Best model not found, using final model instead")

# Load the saved weights
checkpoint = torch.load(model_path, map_location=device, weights_only=False)
generator.load_state_dict(checkpoint['generator_state_dict'])
generator.eval()  # Set to evaluation mode (disables dropout, etc.)

# Display model info if available
if 'ssim' in checkpoint:
    print(f"Model validation SSIM: {checkpoint['ssim']:.4f}")
if 'epoch' in checkpoint:
    print(f"Trained for {checkpoint['epoch']+1} epochs")

print(f"Model loaded from: {model_path}")


# ============================================================
# LOAD TRAINING HISTORY (if available)
# ============================================================

history_path = 'outputs/training_history.json'
if os.path.exists(history_path):
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    print(f"\nTraining History Summary:")
    print(f"  Total epochs: {len(history['epochs'])}")
    print(f"  Final G Loss: {history['g_loss'][-1]:.4f}")
    print(f"  Final C Loss: {history['c_loss'][-1]:.4f}")
    if history['val_ssim']:
        print(f"  Best Val SSIM: {max(history['val_ssim']):.4f}")


# ============================================================
# HELPER FUNCTION: Create image grid for visualization
# ============================================================

def make_grid(images, nrow=8):
    """
    Arrange multiple images into a grid for easy visualization.
    
    Args:
        images: Batch of images (batch_size, channels, height, width)
        nrow: Number of images per row in the grid
    
    Returns:
        Single image tensor containing the grid
    """
    batch_size, channels, h, w = images.shape
    ncol = nrow
    nrow_actual = math.ceil(batch_size / ncol)
    
    # Create white canvas with padding between images
    padding = 2
    grid_h = nrow_actual * h + (nrow_actual + 1) * padding
    grid_w = ncol * w + (ncol + 1) * padding
    grid = torch.ones((channels, grid_h, grid_w), dtype=images.dtype)
    
    # Place each image in the grid
    for idx in range(batch_size):
        row = idx // ncol
        col = idx % ncol
        
        y = row * h + (row + 1) * padding
        x = col * w + (col + 1) * padding
        
        grid[:, y:y+h, x:x+w] = images[idx]
    
    return grid


# ============================================================
# GENERATE RANDOM SAMPLES
# ============================================================

print("\n" + "=" * 60)
print("GENERATING RANDOM SAMPLES")
print("=" * 60)

num_samples = 64  # Generate a grid of 64 images
z = torch.randn(num_samples, latent_dim, device=device)  # Random noise vectors

# Generate images (no gradient computation needed for inference)
with torch.no_grad():
    generated_images = generator(z)

print(f"Generated {num_samples} images")
print(f"Image shape: {generated_images.shape}")
print(f"Image range: [{generated_images.min():.3f}, {generated_images.max():.3f}]")

# Convert from [-1, 1] to [0, 1] range for visualization
generated_images_denorm = denormalize(generated_images)

# Arrange images in an 8x8 grid
grid = make_grid(generated_images_denorm.cpu(), nrow=8)
grid_np = grid.numpy().transpose(1, 2, 0)  # Convert to numpy for matplotlib

# Save the grid visualization
plt.figure(figsize=(16, 16))
plt.imshow(grid_np, cmap='gray')
plt.axis('off')
plt.title('Generated Prostate MRI Samples (64 images)', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig('outputs/generated_samples_grid.png', dpi=300, bbox_inches='tight')
print(f"Saved grid visualization to: outputs/generated_samples_grid.png")
plt.close()


# ============================================================
# LATENT SPACE INTERPOLATION
# ============================================================

print("\n" + "=" * 60)
print("LATENT SPACE INTERPOLATION")
print("=" * 60)

# Create two random starting points in latent space
z1 = torch.randn(1, latent_dim, device=device)
z2 = torch.randn(1, latent_dim, device=device)

# Generate smooth transitions between these two points
num_steps = 10
interpolated_images = []

with torch.no_grad():
    for alpha in np.linspace(0, 1, num_steps):
        # Linear interpolation: z = (1-alpha)*z1 + alpha*z2
        z_interp = alpha * z1 + (1 - alpha) * z2
        img = generator(z_interp)
        interpolated_images.append(img)

# Combine all interpolated images
interpolated_images = torch.cat(interpolated_images, dim=0)
print(f"Created {num_steps} interpolated images")

# Visualize the smooth transition
interp_denorm = denormalize(interpolated_images)
grid_interp = make_grid(interp_denorm.cpu(), nrow=num_steps)
grid_interp_np = grid_interp.numpy().transpose(1, 2, 0)

plt.figure(figsize=(20, 4))
plt.imshow(grid_interp_np, cmap='gray')
plt.axis('off')
plt.title('Latent Space Interpolation (smooth morphing between two random images)', 
          fontsize=14, pad=15)
plt.tight_layout()
plt.savefig('outputs/latent_interpolation.png', dpi=300, bbox_inches='tight')
print(f"Saved interpolation to: outputs/latent_interpolation.png")
plt.close()


# ============================================================
# COMPARISON WITH REAL IMAGES
# ============================================================

print("\n" + "=" * 60)
print("COMPARISON WITH REAL IMAGES")
print("=" * 60)

# Load real test images for comparison
base_dir = "/home/Student/s4743593/hipmri-gan-47435933/keras_slices_data"
test_loader = get_loader(
    os.path.join(base_dir, "keras_slices_test"),
    batch_size=16,
    img_size=img_size,
    shuffle=True,
    num_workers=1,
    augment=False  # No augmentation for testing
)

# Get a batch of real images
real_images = next(iter(test_loader)).to(device)
print(f"Loaded {real_images.shape[0]} real test images")

# Generate the same number of fake images
z_compare = torch.randn(real_images.shape[0], latent_dim, device=device)
with torch.no_grad():
    fake_images = generator(z_compare)

# Compute SSIM score between real and fake images
ssim_scores = []
for i in range(real_images.shape[0]):
    ssim = compute_ssim(fake_images[i:i+1], real_images[i:i+1])
    ssim_scores.append(ssim.item())

avg_ssim = np.mean(ssim_scores)
print(f"\nSSIM Statistics:")
print(f"  Mean: {avg_ssim:.4f}")
print(f"  Std:  {np.std(ssim_scores):.4f}")
print(f"  Min:  {min(ssim_scores):.4f}")
print(f"  Max:  {max(ssim_scores):.4f}")
print(f"  Target threshold (0.6): {'ACHIEVED ✓' if avg_ssim >= 0.6 else 'NOT ACHIEVED ✗'}")

# Create side-by-side comparison
fig, axes = plt.subplots(2, 8, figsize=(20, 6))

real_denorm = denormalize(real_images[:8])
fake_denorm = denormalize(fake_images[:8])

# Top row: real images
for i in range(8):
    axes[0, i].imshow(real_denorm[i].cpu().squeeze(), cmap='gray')
    axes[0, i].axis('off')
    if i == 0:
        axes[0, i].set_title('Real', fontsize=12, pad=10)

# Bottom row: generated images
for i in range(8):
    axes[1, i].imshow(fake_denorm[i].cpu().squeeze(), cmap='gray')
    axes[1, i].axis('off')
    if i == 0:
        axes[1, i].set_title('Generated', fontsize=12, pad=10)

plt.suptitle('Real vs Generated Prostate MRI Images', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('outputs/real_vs_generated.png', dpi=300, bbox_inches='tight')
print(f"Saved comparison to: outputs/real_vs_generated.png")
plt.close()


# ============================================================
# SAVE HIGH-QUALITY INDIVIDUAL SAMPLES
# ============================================================

print("\n" + "=" * 60)
print("SAVING HIGH-QUALITY INDIVIDUAL SAMPLES")
print("=" * 60)

os.makedirs('outputs/individual_samples', exist_ok=True)

num_individual = 10
z_individual = torch.randn(num_individual, latent_dim, device=device)

with torch.no_grad():
    individual_images = generator(z_individual)

individual_denorm = denormalize(individual_images)

# Save each image separately for detailed inspection
for i in range(num_individual):
    img_np = individual_denorm[i].cpu().squeeze().numpy()
    
    plt.figure(figsize=(6, 6))
    plt.imshow(img_np, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'outputs/individual_samples/sample_{i+1:02d}.png', 
                dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()

print(f"Saved {num_individual} individual samples to: outputs/individual_samples/")


# ============================================================
# INTENSITY STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("GENERATING STATISTICS VISUALIZATION")
print("=" * 60)

# Generate many samples to analyze distribution
num_stat_samples = 200
z_stats = torch.randn(num_stat_samples, latent_dim, device=device)

with torch.no_grad():
    stat_images = generator(z_stats)

stat_images_denorm = denormalize(stat_images)

# Flatten all pixel intensities for histogram
intensities = stat_images_denorm.cpu().numpy().flatten()

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram of pixel intensities
axes[0].hist(intensities, bins=100, alpha=0.7, color='blue', edgecolor='black')
axes[0].set_xlabel('Pixel Intensity', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Generated Image Intensity Distribution', fontsize=14)
axes[0].grid(True, alpha=0.3)

# Statistics summary
stats_text = f"""
Generated Images Statistics
{'='*30}
Total samples: {num_stat_samples}
Total pixels: {len(intensities):,}

Intensity Statistics:
  Mean: {np.mean(intensities):.4f}
  Std:  {np.std(intensities):.4f}
  Min:  {np.min(intensities):.4f}
  Max:  {np.max(intensities):.4f}
  
Percentiles:
  25th: {np.percentile(intensities, 25):.4f}
  50th: {np.percentile(intensities, 50):.4f}
  75th: {np.percentile(intensities, 75):.4f}
"""

axes[1].text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', 
             facecolor='wheat', alpha=0.5))
axes[1].axis('off')

plt.tight_layout()
plt.savefig('outputs/intensity_statistics.png', dpi=300, bbox_inches='tight')
print(f"Saved statistics to: outputs/intensity_statistics.png")
plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PREDICTION COMPLETE - SUMMARY")
print("=" * 60)

print(f"\nGenerated outputs:")
print(f"  1. Random samples grid: outputs/generated_samples_grid.png")
print(f"  2. Latent interpolation: outputs/latent_interpolation.png")
print(f"  3. Real vs Generated: outputs/real_vs_generated.png")
print(f"  4. Individual samples: outputs/individual_samples/ (10 images)")
print(f"  5. Intensity statistics: outputs/intensity_statistics.png")

print(f"\nModel Performance:")
print(f"  Average SSIM: {avg_ssim:.4f}")
print(f"  SSIM threshold (0.6): {'✓ ACHIEVED' if avg_ssim >= 0.6 else '✗ NOT ACHIEVED'}")
print(f"  Image quality: {'Reasonably clear' if avg_ssim >= 0.6 else 'Needs improvement'}")

print("\n" + "=" * 60)
print("All visualizations saved successfully!")
print("=" * 60)
