# train.py (WGAN-GP version)
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
lr = 1e-4
n_epochs = 100
n_critic = 5
lambda_gp = 10
device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------
# Data loaders
# ----------------------
base_dir = "/home/Student/s4743593/hipmri-gan-47435933/keras_slices_data"
train_loader = get_loader(os.path.join(base_dir, "keras_slices_train"), batch_size=batch_size, img_size=img_size)
val_loader   = get_loader(os.path.join(base_dir, "keras_slices_validate"), batch_size=batch_size, img_size=img_size, shuffle=False)
test_loader  = get_loader(os.path.join(base_dir, "keras_slices_test"), batch_size=batch_size, img_size=img_size, shuffle=False)

# ----------------------
# Model setup
# ----------------------
G = Generator(latent_dim=latent_dim).to(device)
D = Discriminator().to(device)

optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.0, 0.9))
optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.0, 0.9))

# ----------------------
# Training preparation
# ----------------------
G_losses, D_losses, SSIM_scores = [], [], []
val_ssims, test_ssims = [], []

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
            z = torch.randn(batch_size_curr, latent_dim, 1, 1).to(device)
            fake_imgs = G(z).detach()
            
            D_real = D(imgs)
            D_fake = D(fake_imgs)
            
            # Wasserstein loss
            loss_D = -(torch.mean(D_real) - torch.mean(D_fake))
            
            # Gradient penalty
            epsilon = torch.rand(batch_size_curr, 1, 1, 1, device=device)
            x_hat = epsilon * imgs + (1 - epsilon) * fake_imgs
            x_hat.requires_grad_(True)
            D_hat = D(x_hat)
            
            gradients = torch.autograd.grad(
                outputs=D_hat, inputs=x_hat,
                grad_outputs=torch.ones_like(D_hat),
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            gradients = gradients.view(batch_size_curr, -1)
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
            
            # Total discriminator loss
            loss_D_total = loss_D + lambda_gp * gradient_penalty

            optimizer_D.zero_grad()
            loss_D_total.backward()
            optimizer_D.step()

        # ---------------------
        # Train Generator
        # ---------------------
        z = torch.randn(batch_size_curr, latent_dim, 1, 1).to(device)
        gen_imgs = G(z)
        loss_G = -torch.mean(D(gen_imgs))

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        # Record losses & SSIM
        if i % 100 == 0:
            G_losses.append(loss_G.item())
            D_losses.append(loss_D_total.item())
            
            with torch.no_grad():
                ssim_score = ssim((gen_imgs + 1) / 2, (imgs + 1) / 2)
                SSIM_scores.append(ssim_score.item())

            print(f"[Epoch {epoch}/{n_epochs}] [Batch {i}/{len(train_loader)}] "
                  f"D: {loss_D_total.item():.4f}, G: {loss_G.item():.4f}, SSIM: {ssim_score.item():.4f}")

    # ---------------------
    # Validation
    # ---------------------
    G.eval()
    with torch.no_grad():
        val_ssim_sum = 0
        for imgs in val_loader:
            imgs = imgs.to(device)
            z = torch.randn(imgs.size(0), latent_dim, 1, 1).to(device)
            gen_imgs = G(z)
            val_ssim_sum += ssim((gen_imgs + 1) / 2, (imgs + 1) / 2).item()
        val_avg_ssim = val_ssim_sum / len(val_loader)
        val_ssims.append(val_avg_ssim)
        print(f"Validation SSIM after Epoch {epoch}: {val_avg_ssim:.4f}")

    # ---------------------
    # Save generated samples
    # ---------------------
    with torch.no_grad():
        z = torch.randn(16, latent_dim, 1, 1).to(device)
        samples = G(z)
        samples = (samples + 1) / 2
        save_image(samples, f"generated_samples/epoch_{epoch}.png", nrow=4)

    # ---------------------
    # Save models every 10 epochs
    # ---------------------
    if epoch % 10 == 0:
        torch.save(G.state_dict(), f"saved_models/G_epoch_{epoch}.pth")
        torch.save(D.state_dict(), f"saved_models/D_epoch_{epoch}.pth")

# ----------------------
# Testing phase
# ----------------------
G.eval()
with torch.no_grad():
    test_ssim_sum = 0
    for imgs in test_loader:
        imgs = imgs.to(device)
        z = torch.randn(imgs.size(0), latent_dim, 1, 1).to(device)
        gen_imgs = G(z)
        test_ssim_sum += ssim((gen_imgs + 1) / 2, (imgs + 1) / 2).item()
    test_avg_ssim = test_ssim_sum / len(test_loader)
    test_ssims.append(test_avg_ssim)
    print(f"\nFinal Test SSIM: {test_avg_ssim:.4f}")

# ----------------------
# Plot Losses & SSIM
# ----------------------
plt.figure(figsize=(10,5))
plt.plot(G_losses, label="Generator")
plt.plot(D_losses, label="Discriminator")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("generated_samples/loss_curve.png")
plt.close()

plt.figure()
plt.plot(SSIM_scores, label="Train SSIM (per 100 iters)")
plt.plot(val_ssims, label="Validation SSIM (per epoch)")
plt.xlabel("Iterations/Epochs")
plt.ylabel("SSIM")
plt.legend()
plt.tight_layout()
plt.savefig("generated_samples/ssim_curve.png")
plt.close()

