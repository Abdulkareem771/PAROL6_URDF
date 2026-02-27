import os
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from pathlib import Path


current_dir = Path(__file__)
project_dir = current_dir.parent.parent

IMAGE_DIR = project_dir / "data" / "U-Net_data" / "U-Net_model_v2 (modified)" / "dataset" / "images"
MASK_DIR = project_dir / "data" / "U-Net_data" / "U-Net_model_v2 (modified)" / "dataset" / "masks"


# Patch-Based Dataset (Memory-Safe)

class SeamPatchDataset(Dataset):
    def __init__(self, image_dir, mask_dir, patch_size=384, stride=320):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))
        self.patch_size = patch_size
        self.stride = stride

        self.to_tensor = T.ToTensor()

        # Precompute patch coordinates
        self.samples = []
        for img_name in self.images:
            img_path = os.path.join(image_dir, img_name)
            img = Image.open(img_path)
            w, h = img.size

            for y in range(0, h - patch_size + 1, stride):
                for x in range(0, w - patch_size + 1, stride):
                    self.samples.append((img_name, x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, x, y = self.samples[idx]

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = image.crop((x, y, x + self.patch_size, y + self.patch_size))
        mask = mask.crop((x, y, x + self.patch_size, y + self.patch_size))

        image = self.to_tensor(image)
        mask = self.to_tensor(mask)
        mask = (mask > 0.5).float()  # binarize

        return image, mask


# CPU-Friendly U-Net (Thin-Seam Safe)

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.down1 = DoubleConv(3, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(64, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv1 = DoubleConv(64, 32)

        self.final = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        bn = self.bottleneck(self.pool2(d2))

        u2 = self.up2(bn)
        u2 = self.conv2(torch.cat([u2, d2], dim=1))

        u1 = self.up1(u2)
        u1 = self.conv1(torch.cat([u1, d1], dim=1))

        return self.final(u1)


# Dice + BCE Loss (Correct & Stable)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )
        return 1 - dice

bce_loss = nn.BCEWithLogitsLoss()
dice_loss = DiceLoss()

# Training & Validation Loop (CPU-Safe)

def train():
    DEVICE = "cpu"
    EPOCHS = 50
    BATCH_SIZE = 1
    LR = 1e-4

    dataset = SeamPatchDataset(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR
    )

    val_split = int(0.8 * len(dataset))
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [val_split, len(dataset) - val_split]
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=0
    )

    model = UNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            preds = model(imgs)
            loss = bce_loss(preds, masks) + dice_loss(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ---- Validation ----
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                preds = model(imgs)
                loss = bce_loss(preds, masks) + dice_loss(preds, masks)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch [{epoch+1}/{EPOCHS}] | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_seam_unet.pth")
            print("✔ Best model saved")

    print("Training complete.")


# Run Training

if __name__ == "__main__":
    train()
        