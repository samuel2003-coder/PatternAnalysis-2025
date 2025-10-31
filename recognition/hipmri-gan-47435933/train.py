"""
Training script for WGAN-GP on HipMRI prostate cancer dataset
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

from modules import Generator, Critic, compute_gradient_penalty, weights_init, compute_ssim
from dataset import get_loader, save_image_grid, denormalize


# ----------------------
# Hyperparameters
# ----------------------
latent_dim = 128 # Size of random noise vector  
img_size = 128 # Size of generated image
img_channels = 1 # Channels in input images (always 1 for grayscale)
batch_size = 16  # Samples per iteration
n_epochs = 300  # Number of iterations
lr_g = 0.0001  # Separate, lower learning rate for generator
lr_c = 0.0004  # Higher learning rate for critic
# Variables used to control the Adam optimiser gradient updates
b1 = 0.0
b2 = 0.9  
n_critic = 5  # Train critic n_critic times per generator update
lambda_gp = 10  # Gradient penalty weight
sample_interval = 500  # Save samples every N batches

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create directories
os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/samples", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# Data loaders
base_dir = "/home/Student/s4743593/hipmri-gan-47435933/keras_slices_data"
train_loader = get_loader(os.path.join(base_dir, "keras_slices_train"), batch_size=batch_size, img_size=img_size, augment=True, num_workers=1)
val_loader = get_loader(os.path.join(base_dir, "keras_slices_validate"), batch_size=batch_size, img_size=img_size, shuffle=False, num_workers=1)
test_loader = get_loader(os.path.join(base_dir, "keras_slices_test"), batch_size=batch_size, img_size=img_size, shuffle=False, num_workers=1)

print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")

# Initialize models
generator = Generator(latent_dim=latent_dim, img_channels=img_channels).to(device)
critic = Critic(img_channels=img_channels).to(device)

# Initialize weights
generator.apply(weights_init)
critic.apply(weights_init)

print(f"\nGenerator parameters: {sum(p.numel() for p in generator.parameters()):,}")
print(f"Critic parameters: {sum(p.numel() for p in critic.parameters()):,}")

# Optimizers, note the use of separate learning rates
optimizer_G = optim.Adam(generator.parameters(), lr=lr_g, betas=(b1, b2))
optimizer_C = optim.Adam(critic.parameters(), lr=lr_c, betas=(b1, b2))

# Training history used for tracking progress
history = {
    'g_loss': [],
    'c_loss': [],
    'wasserstein_dist': [],
    'val_ssim': [],
    'epochs': []
}

# Fixed noise for visualisation
fixed_z = torch.randn(64, latent_dim, device=device)


def validate(generator, val_loader, device, num_samples=500):
    """
    Validate the generator using SSIM metric
    
    Args:
        generator: Generator model
        val_loader: Validation data loader
        device: torch device
        num_samples: Number of samples to use for validation
    Returns:
        Average SSIM score
    """
    generator.eval()
    ssim_scores = []
    
    # Iterate over the validation set
    with torch.no_grad():
        for i, real_imgs in enumerate(val_loader):
            if i * batch_size >= num_samples:
                break
            
            real_imgs = real_imgs.to(device)
            batch_size_actual = real_imgs.size(0)
            
            # Generate fake images
            z = torch.randn(batch_size_actual, latent_dim, device=device)
            fake_imgs = generator(z)
            
            # Compute SSIM between real and fake images
            # Compare each fake image with a random real image
            for j in range(batch_size_actual):
                real_idx = np.random.randint(0, batch_size_actual)
                ssim = compute_ssim(
                    fake_imgs[j:j+1],
                    real_imgs[real_idx:real_idx+1]
                )
                # Append current SSIM score calculated to a ssim_scores list
                ssim_scores.append(ssim.item())
    
    generator.train()
    # Return the mean SSIM score for this batch
    return np.mean(ssim_scores)


def test_model(generator, test_loader, device, save_dir="outputs/test_results"):
    """
    Test the trained generator
    
    Args:
        generator: Trained generator model
        test_loader: Test data loader
        device: torch device
        save_dir: Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)
    # Set the model to evaluation mode
    generator.eval()
    
    # Formatting for output clarity
    print("\n" + "="*50)
    print("TESTING GENERATOR")
    print("="*50)
    
    # Generate samples and compute SSIM
    ssim_scores = []
    num_test_samples = min(500, len(test_loader) * batch_size)
    
    # Iterate through the testing set
    with torch.no_grad():
        for i, real_imgs in enumerate(tqdm(test_loader, desc="Testing")):
            if i * batch_size >= num_test_samples:
                break
            
            real_imgs = real_imgs.to(device)
            batch_size_actual = real_imgs.size(0)
            
            # Generate fake images
            z = torch.randn(batch_size_actual, latent_dim, device=device)
            fake_imgs = generator(z)
            
            # Compute SSIM between real and fake images for each image in batch
            for j in range(batch_size_actual):
                real_idx = np.random.randint(0, batch_size_actual)
                ssim = compute_ssim(
                    fake_imgs[j:j+1],
                    real_imgs[real_idx:real_idx+1]
                )
                # Append current SSIM score calculated to a ssim_scores list
                ssim_scores.append(ssim.item())
            
            # Save first batch for visualization
            if i == 0:
                save_image_grid(
                    fake_imgs[:16],
                    os.path.join(save_dir, "generated_samples.png"),
                    nrow=4
                )
                save_image_grid(
                    real_imgs[:16],
                    os.path.join(save_dir, "real_samples.png"),
                    nrow=4
                )
    
    # Keep track of mean and standard deviation of SSIM scores
    avg_ssim = np.mean(ssim_scores)
    std_ssim = np.std(ssim_scores)
    
    # Output results
    print(f"\nTest Results:")
    print(f"  Average SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}")
    print(f"  Samples tested: {len(ssim_scores)}")
    print(f"  Images saved to: {save_dir}")
    
    # Save test results
    test_results = {
        'avg_ssim': float(avg_ssim),
        'std_ssim': float(std_ssim),
        'num_samples': len(ssim_scores),
        'all_ssim_scores': [float(s) for s in ssim_scores]
    }
    
    with open(os.path.join(save_dir, "test_results.json"), 'w') as f:
        json.dump(test_results, f, indent=4)
    
    return avg_ssim


# Training loop

# Formatting for output clarity
print("\n" + "="*50)
print("STARTING TRAINING")
print("="*50 + "\n")

# Keep track of how far into the training we are and the progress by recording the best SSIM so far
batches_done = 0
best_ssim = 0.0


# Iterate through epochs
for epoch in range(n_epochs):
    # Keep track of generator and critic loss, and Wassertain distances
    epoch_g_loss = []
    epoch_c_loss = []
    epoch_wd = []
    
    print(f"\n{'='*60}")
    print(f"Epoch {epoch+1}/{n_epochs}")
    print(f"{'='*60}")
    
    # Iterate through training set
    for i, real_imgs in enumerate(train_loader):
        real_imgs = real_imgs.to(device)
        batch_size_actual = real_imgs.size(0)
        
        # Train Critic
        # Set critic gradients to 0
        optimizer_C.zero_grad()
        
        # Generate fake images
        z = torch.randn(batch_size_actual, latent_dim, device=device)
        fake_imgs = generator(z).detach()
        
        # Critic scores
        real_validity = critic(real_imgs)
        fake_validity = critic(fake_imgs)
        
        # Compute gradient penalty for critic scores
        gradient_penalty = compute_gradient_penalty(
            critic, real_imgs, fake_imgs, device
        )
        
        # Critic loss (note the penalty being included)
        c_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
        
        # Perform backpropagation and compute gradients
        c_loss.backward()
        optimizer_C.step()
        
        # Append current critic loss value to epoch_c_loss list
        epoch_c_loss.append(c_loss.item())
        
        # Wasserstein distance calculated by using the mean scores for real and fake images
        wd = torch.mean(real_validity) - torch.mean(fake_validity)
        # Recorded by appending to epoch_wd list
        epoch_wd.append(wd.item())
        
        # Train Generator
        if i % n_critic == 0:
            # Reset generator gradients to 0 since we have backpropagated
            optimizer_G.zero_grad()
            
            # Generate fake images
            z = torch.randn(batch_size_actual, latent_dim, device=device)
            gen_imgs = generator(z)
            
            # Generator loss (Wasserstein)
            fake_validity = critic(gen_imgs)
            g_loss_adv = -torch.mean(fake_validity)
            
            # Add adversarial loss for better quality
            g_loss = g_loss_adv
            
            # Perform backpropagation and compute gradients
            g_loss.backward()
            optimizer_G.step()
            
            # Keep track of geneerator losses by adding to epoch_g_loss list
            epoch_g_loss.append(g_loss.item())
        
        # Progress update - every 50 batches
        if i % 50 == 0 and len(epoch_g_loss) > 0:
            print(f"  [{i:4d}/{len(train_loader)}] G: {epoch_g_loss[-1]:7.4f} | "
                  f"C: {c_loss.item():7.4f} | WD: {wd.item():7.4f}")
        
        # Save samples
        if batches_done % sample_interval == 0:
            with torch.no_grad():
                gen_imgs = generator(fixed_z)
                # Save generator images in outputs/samples
                save_image_grid(
                    gen_imgs,
                    f"outputs/samples/epoch_{epoch}_batch_{batches_done}.png",
                    nrow=8
                )
            if batches_done > 0:
                print(f"  Saved sample images at batch {batches_done}")
        
        batches_done += 1
    
    # Epoch summary displayed on output
    # Mean generator and critic losses and mean wasserstein distance
    avg_g_loss = np.mean(epoch_g_loss) if epoch_g_loss else 0
    avg_c_loss = np.mean(epoch_c_loss)
    avg_wd = np.mean(epoch_wd)
    
    history['g_loss'].append(avg_g_loss)
    history['c_loss'].append(avg_c_loss)
    history['wasserstein_dist'].append(avg_wd)
    history['epochs'].append(epoch + 1)
    
    # Print epoch summary
    print(f"\n{'='*60}")
    print(f"Epoch {epoch+1}/{n_epochs} Summary:")
    print(f"  Generator Loss    : {avg_g_loss:7.4f}")
    print(f"  Critic Loss       : {avg_c_loss:7.4f}")
    print(f"  Wasserstein Dist  : {avg_wd:7.4f}")
    
    # Quick SSIM check every epoch (fewer samples for speed)
    quick_ssim = validate(generator, val_loader, device, num_samples=100)
    print(f"  Quick SSIM Check  : {quick_ssim:.4f}")
    
    # Validation
    if (epoch + 1) % 5 == 0:
        print(f"\n{'─'*60}")
        print(f"Running validation...")
        val_ssim = validate(generator, val_loader, device)
        history['val_ssim'].append(val_ssim)
        print(f"  Validation SSIM   : {val_ssim:.4f}")
        print(f"  Target SSIM       : 0.6000")
        print(f"  Status            : {'✓ ACHIEVED' if val_ssim >= 0.6 else '✗ Below target'}")
        print(f"{'─'*60}")
        
        # Save best model
        if val_ssim > best_ssim:
            best_ssim = val_ssim
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'critic_state_dict': critic.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_C_state_dict': optimizer_C.state_dict(),
                'ssim': val_ssim,
            }, 'checkpoints/best_model.pth')
            print(f"  ★ Best model saved with SSIM: {val_ssim:.4f}")
    else:
        # Show last known SSIM even on non-validation epochs
        if history['val_ssim']:
            print(f"  Last SSIM: {history['val_ssim'][-1]:.4f} (Best: {best_ssim:.4f})")
    
    print(f"{'='*60}\n")
    
    # Save checkpoint every 20 epochs
    if (epoch + 1) % 20 == 0:
        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'critic_state_dict': critic.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_C_state_dict': optimizer_C.state_dict(),
        }, f'checkpoints/checkpoint_epoch_{epoch+1}.pth')
        print(f"Checkpoint saved at epoch {epoch+1}\n")

# ----------------------
# Final model save
# ----------------------
torch.save({
    'generator_state_dict': generator.state_dict(),
    'critic_state_dict': critic.state_dict(),
}, 'checkpoints/final_model.pth')

# Save training history
with open('outputs/training_history.json', 'w') as f:
    json.dump(history, f, indent=4)

# ----------------------
# Plot training curves
# ----------------------
print("\nGenerating training plots...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Generator and Critic loss
axes[0, 0].plot(history['epochs'], history['g_loss'], label='Generator Loss', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Generator Loss')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

axes[0, 1].plot(history['epochs'], history['c_loss'], label='Critic Loss', linewidth=2, color='orange')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].set_title('Critic Loss')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

# Wasserstein distance
axes[1, 0].plot(history['epochs'], history['wasserstein_dist'], label='Wasserstein Distance', linewidth=2, color='green')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Distance')
axes[1, 0].set_title('Wasserstein Distance')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()

# Validation SSIM
if history['val_ssim']:
    val_epochs = [e for e in history['epochs'] if e % 5 == 0]
    axes[1, 1].plot(val_epochs, history['val_ssim'], label='Validation SSIM', linewidth=2, color='red', marker='o')
    axes[1, 1].axhline(y=0.6, color='black', linestyle='--', label='Target SSIM (0.6)', alpha=0.5)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('SSIM')
    axes[1, 1].set_title('Validation SSIM')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

plt.tight_layout()
plt.savefig('outputs/training_curves.png', dpi=300, bbox_inches='tight')
print(f"Training curves saved to outputs/training_curves.png")

# Test the model
print("\n" + "="*50)
print("Testing best model...")
print("="*50)

# Load best model
checkpoint = torch.load('checkpoints/best_model.pth')
generator.load_state_dict(checkpoint['generator_state_dict'])

# Test
test_ssim = test_model(generator, test_loader, device)

print("\n" + "="*50)
print("TRAINING COMPLETE")
print("="*50)
print(f"Best Validation SSIM: {best_ssim:.4f}")
print(f"Test SSIM: {test_ssim:.4f}")
print(f"Target SSIM threshold: 0.6")
print(f"Target achieved: {'YES' if test_ssim >= 0.6 else 'NO'}")
