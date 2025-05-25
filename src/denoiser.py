import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import deeplay as dl

class GrayDataset(Dataset):
    """
    Loads images from a directory (and subdirectories) as single-channel grayscale tensors.
    Uses os.walk for directory traversal.
    """
    def __init__(self, root_dir, extensions=('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
        self.paths = []
        for dirpath, _, filenames in os.walk(root_dir):
            for fname in filenames:
                if fname.lower().endswith(extensions):
                    self.paths.append(os.path.join(dirpath, fname))
        self.paths.sort()
        if not self.paths:
            raise FileNotFoundError(f"No images found in {root_dir} with extensions {extensions}")

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        with Image.open(img_path) as img:
            img = img.convert('RGB')
            clean = self.transform(img)
        return clean

class NoisyGrayDataset(Dataset):
    """
    Wraps GrayDataset and returns (noisy, clean) grayscale tensors.
    """
    def __init__(self, root_dir, noise_std=0.3, extensions=('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
        self.clean_ds = GrayDataset(root_dir, extensions)
        self.noise_std = noise_std

    def __len__(self):
        return len(self.clean_ds)

    def __getitem__(self, idx):
        clean = self.clean_ds[idx]
        noise = torch.randn_like(clean) * self.noise_std
        noisy = torch.clamp(clean + noise, 0.0, 1.0)
        return noisy, clean

def test(model):
    root_data_dir = "test"
    batch_size = 32
    noise_std = 0.1

    test_set = NoisyGrayDataset(root_data_dir, noise_std=noise_std)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)

    noisy_imgs, clean_imgs = next(iter(test_loader))
    noisy_5, clean_5 = noisy_imgs[:5], clean_imgs[:5]

    fig, axes = plt.subplots(5, 2, figsize=(6, 15))
    for i in range(5):
        axes[i, 0].imshow(clean_5[i].squeeze().cpu().numpy(), cmap='gray')
        axes[i, 0].set_title(f'Clean #{i+1}')
        axes[i, 0].axis('off')
        axes[i, 1].imshow(noisy_5[i].squeeze().cpu().numpy(), cmap='gray')
        axes[i, 1].set_title(f'Noisy #{i+1}')
        axes[i, 1].axis('off')
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(5, 3, figsize=(10, 15))
    for i in range(5):
        input_img = noisy_5[i]
        target_img = clean_5[i]
        pred_img = model(input_img.unsqueeze(0)).detach()
        axes[i, 0].imshow(input_img[0].cpu().numpy(), cmap='gray')
        axes[i, 0].set_title(f"Input {i+1}")
        axes[i, 0].axis('off')
        axes[i, 1].imshow(target_img[0].cpu().numpy(), cmap='gray')
        axes[i, 1].set_title(f"Target {i+1}")
        axes[i, 1].axis('off')
        axes[i, 2].imshow(pred_img[0, 0].cpu().numpy(), cmap='gray')
        axes[i, 2].set_title(f"Predicted {i+1}")
        axes[i, 2].axis('off')
    plt.tight_layout()
    plt.show()

def main():
    root_data_dir = "dataset"
    batch_size = 64
    noise_std = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = NoisyGrayDataset(root_data_dir, noise_std=noise_std)
   
    fraction = 1
    num_samples = int(len(dataset) * fraction)
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    subset = Subset(dataset, indices)

    loader = dl.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    print(f"Subset size: {len(subset)}")

    unet = dl.UNet2d(
        in_channels=1, channels=[16, 32, 64, 128, 256], out_channels=1, skip=dl.Cat(),
    )

    if torch.cuda.is_available():
        print("GPU model:", torch.cuda.get_device_name(0))
    else:
        print("No GPU available")

    regressor_template = dl.Regressor(
        model=unet, loss=nn.MSELoss(), optimizer=dl.Adam(lr=1e-3),
    )
    noise_model = regressor_template.create()
    ed_trainer = dl.Trainer(max_epochs=200, accelerator="cuda", devices=1)
    print("start")
    ed_trainer.fit(noise_model, loader)
    test(noise_model)
   
    torch.save(noise_model.state_dict(), "denoiser_model.pth")
    print("Model saved as denoiser_model_0111.pth")

if __name__ == "__main__":
    main()