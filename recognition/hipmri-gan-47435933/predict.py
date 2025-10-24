# predict.py
import torch
from modules import Generator
import matplotlib.pyplot as plt

latent_dim = 100
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load generator
G = Generator(latent_dim=latent_dim).to(device)
G.load_state_dict(torch.load("saved_models/G.pth", map_location=device))
G.eval()

# Generate images
z = torch.randn(16, latent_dim, 1, 1).to(device)
with torch.no_grad():
    gen_imgs = G(z).cpu()

# Plot
fig, axs = plt.subplots(4, 4, figsize=(8,8))
for i, ax in enumerate(axs.flatten()):
    ax.imshow(gen_imgs[i][0]*0.5 + 0.5, cmap="gray")
    ax.axis("off")
plt.show()

