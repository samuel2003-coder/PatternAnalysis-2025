# train.py
import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torchmetrics.functional import structural_similarity_index_measure as ssim

from modules import Generator, Discriminator
from dataset import get_loader

# ----------------------
# Hyperparameters
# ----------------------
latent_dim = 100
img_size = 64
batch_size = 32
lr = 0.00005
n_epochs = 100
n_critic = 5
clip_value = 0.01
device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------
# Data loaders
# ----------------------
# Data loaders for the different datasets by calling the helper function in dataset.py
# Preprocessing the data by converting to tensors and normalising
base_dir = r"C:\Users\samue\Documents\Semester 2 2025\COMP3710\PatternAnalysis-2025\recognition\keras_slices_data"

train_loader = get_loader(os.path.join(base_dir, "keras_slices_train"), batch_size=batch_size, img_size=img_size)
val_loader   = get_loader(os.path.join(base_dir, "keras_slices_validate"), batch_size=batch_size, img_size=img_size, shuffle=False)
test_loader  = get_loader(os.path.join(base_dir, "keras_slices_test"), batch_size=batch_size, img_size=img_size, shuffle=False)

# ----------------------
# Model setup
# ----------------------
# Defining generator and discriminator models through the defined classes in modules.py
G = Generator(latent_dim=latent_dim).to(device)
D = Discriminator().to(device)

# Using RMSProp optimiser as it performs better and provides more stability than Adam optimiser for WGANs
optimizer_G = optim.RMSprop(G.parameters(), lr=lr)
optimizer_D = optim.RMSprop(D.parameters(), lr=lr)

# ----------------------
# Training preparation
# ----------------------
# Storing generator, discriminator losses and SSIM scores for later plotting
G_losses, D_losses, SSIM_scores = [], [], []

# Track validation and test SSIMs separately
val_ssims, test_ssims = [], []

# Storing generated samples and model weights for optional analysis
os.makedirs("generated_samples", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

# ----------------------
# Training Loop
# ----------------------
for epoch in range(1, n_epochs + 1):
    G.train()
    D.train()

    for i, imgs in enumerate(train_loader):
        imgs = imgs.to(device)
        batch_size_curr = imgs.size(0)

        # ---------------------
        # Train Discriminator
        # ---------------------
        for _ in range(n_critic):
            # Random noise input
            z = torch.randn(batch_size_curr, latent_dim, 1, 1).to(device)
            # Create fake images by calling Generator G on z
            fake_imgs = G(z).detach()
            # Get Discriminator scores for real and fake images
            D_real = D(imgs)
            D_fake = D(fake_imgs)

            # Wasserstein loss for Discriminator
            # Discriminator wants D_real to be high and D_fake to be low
            loss_D = -(torch.mean(D_real) - torch.mean(D_fake))

            # Clear gradients from previous iteration and update loss_D
            optimizer_D.zero_grad()
            loss_D.backward()
            # Update discriminator weights
            optimizer_D.step()

            # Weight clipping (WGAN)
            # Ensures weights don't get too large. Critic score must be 1-Lipschitz.
            for p in D.parameters():
                p.data.clamp_(-clip_value, clip_value)

        # ---------------------
        # Train Generator
        # ---------------------
        # Generate images by calling Generator class
        z = torch.randn(batch_size_curr, latent_dim, 1, 1).to(device)
        gen_imgs = G(z)
        # How real the discriminator thinks the generated images are.
        # Generator wants to maximise this — Wasserstein loss for Generator.
        loss_G = -torch.mean(D(gen_imgs))

        # Update gradients and update generator's parameters
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        # For every 100 batches, record losses and calculate SSIM
        if i % 100 == 0:
            G_losses.append(loss_G.item())
            D_losses.append(loss_D.item())

            # Compute SSIM only on the first batch to reduce cost
            with torch.no_grad():
                # Normalise images for SSIM into [0, 1]
                ssim_score = ssim((gen_imgs + 1) / 2, (imgs + 1) / 2)
                SSIM_scores.append(ssim_score.item())

            print(f"[Epoch {epoch}/{n_epochs}] [Batch {i}/{len(train_loader)}] "
                  f"D: {loss_D.item():.4f}, G: {loss_G.item():.4f}, SSIM: {ssim_score.item():.4f}")

    # ---------------------
    # Validation phase
    # ---------------------
    G.eval()
    with torch.no_grad():
        val_ssim_sum = 0
        # Loop over validation data
        for imgs in val_loader:
            # Real images
            imgs = imgs.to(device)
            # Generated images
            z = torch.randn(imgs.size(0), latent_dim, 1, 1).to(device)
            gen_imgs = G(z)
            # Calculate SSIM between generated and real images
            val_ssim_sum += ssim((gen_imgs + 1) / 2, (imgs + 1) / 2).item()
        # Average SSIM for a batch and store it
        val_avg_ssim = val_ssim_sum / len(val_loader)
        val_ssims.append(val_avg_ssim)
        print(f"Validation SSIM after Epoch {epoch}: {val_avg_ssim:.4f}")

    # ---------------------
    # Save 16 MRI slice images per epoch for optional visualisation
    # ---------------------
    with torch.no_grad():
        z = torch.randn(16, latent_dim, 1, 1).to(device)
        samples = G(z)
        samples = (samples + 1) / 2  # scale from [-1,1] to [0,1]
        save_image(samples, f"generated_samples/epoch_{epoch}.png", nrow=4)

    # Save model weights every few epochs
    if epoch % 10 == 0:
        torch.save(G.state_dict(), f"saved_models/G_epoch_{epoch}.pth")
        torch.save(D.state_dict(), f"saved_models/D_epoch_{epoch}.pth")

# ----------------------
# Testing phase (final evaluation)
# ----------------------
G.eval()
with torch.no_grad():
    test_ssim_sum = 0
    # Loop over test data
    for imgs in test_loader:
        # Real images
        imgs = imgs.to(device)
        z = torch.randn(imgs.size(0), latent_dim, 1, 1).to(device)
        # Fake images
        gen_imgs = G(z)
        test_ssim_sum += ssim((gen_imgs + 1) / 2, (imgs + 1) / 2).item()
    test_avg_ssim = test_ssim_sum / len(test_loader)
    # Average SSIM score per batch
    test_ssims.append(test_avg_ssim)
    print(f"\nFinal Test SSIM: {test_avg_ssim:.4f}")

# ----------------------
# Plot Losses and SSIM
# ----------------------
plt.figure(figsize=(10,5))
plt.plot(G_losses, label="Generator")
plt.plot(D_losses, label="Discriminator")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(SSIM_scores, label="Train SSIM (per 100 iters)")
plt.plot(val_ssims, label="Validation SSIM (per epoch)")
plt.xlabel("Iterations/Epochs")
plt.ylabel("SSIM")
plt.legend()
plt.tight_layout()
plt.show()

