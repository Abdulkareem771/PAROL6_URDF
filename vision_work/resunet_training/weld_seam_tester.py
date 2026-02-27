import argparse
import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# --- ResUNet Definition ---
class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        res = self.shortcut(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x + res

class ResUNet(nn.Module):
    def __init__(self, in_c=3, out_c=1):
        super().__init__()
        self.conv_init = nn.Sequential(
            nn.Conv2d(in_c, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        )
        self.shortcut_init = nn.Sequential(
            nn.Conv2d(in_c, 64, 1, 1, bias=False),
            nn.BatchNorm2d(64)
        )
        self.res1 = ResidualBlock(64, 128, stride=2)
        self.res2 = ResidualBlock(128, 256, stride=2)
        self.res3 = ResidualBlock(256, 512, stride=2)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = ResidualBlock(512, 256) 
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = ResidualBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = ResidualBlock(128, 64)
        self.final_conv = nn.Conv2d(64, out_c, 1)
        
    def forward(self, x):
        res_init = self.shortcut_init(x)
        c1 = self.conv_init(x) + res_init
        c2 = self.res1(c1)
        c3 = self.res2(c2)
        c4 = self.res3(c3)
        u3 = self.up3(c4)
        u3 = torch.cat([u3, c3], dim=1)
        d3 = self.dec3(u3)
        u2 = self.up2(d3)
        u2 = torch.cat([u2, c2], dim=1)
        d2 = self.dec2(u2)
        u1 = self.up1(d2)
        u1 = torch.cat([u1, c1], dim=1)
        d1 = self.dec1(u1)
        out = self.final_conv(d1)
        return out

def test_image(image_path, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from {model_path} on {device}...")
    
    model = ResUNet().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return
    model.eval()

    print(f"Loading image {image_path}...")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}. Check the path.")
        return
        
    original_h, original_w = img.shape[:2]
    
    # Preprocess
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (512, 512))
    
    # Normalize like training pipeline
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_normalized = (img_resized / 255.0 - mean) / std
    
    img_tensor = torch.tensor(img_normalized).float().permute(2, 0, 1).unsqueeze(0).to(device)

    print("Running inference...")
    with torch.no_grad():
        preds = model(img_tensor)
        probs = torch.sigmoid(preds)
        mask_pred = (probs > 0.5).squeeze().cpu().numpy()

    # Resize mask back to original physical resolution
    mask_resized = cv2.resize(mask_pred.astype(np.uint8), (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    
    # Create overlay
    overlay = img.copy()
    overlay[mask_resized == 1] = [0, 0, 255] # Red overlay for the seam
    
    # Skeletonize (extract 1-pixel centerline)
    try:
        from skimage.morphology import skeletonize
        skeleton = skeletonize(mask_resized)
        skeleton_overlay = img.copy()
        skeleton_overlay[skeleton] = [0, 255, 0] # Green centerline
    except ImportError:
        skeleton_overlay = None

    # Save outputs
    os.makedirs("test_results", exist_ok=True)
    base_name = os.path.basename(image_path)
    cv2.imwrite(f"test_results/overlay_{base_name}", overlay)
    print(f"Saved seam overlay to test_results/overlay_{base_name}")
    
    if skeleton_overlay is not None:
        cv2.imwrite(f"test_results/skeleton_{base_name}", skeleton_overlay)
        print(f"Saved skeleton centerline to test_results/skeleton_{base_name}")

    # Display plots
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title("Original ROI")
    
    plt.subplot(1, 3, 2)
    plt.imshow(mask_resized, cmap='gray')
    plt.title("Predicted Seam Mask")
    
    plt.subplot(1, 3, 3)
    if skeleton_overlay is not None:
        plt.imshow(cv2.cvtColor(skeleton_overlay, cv2.COLOR_BGR2RGB))
        plt.title("Skeleton Centerline (Green)")
    else:
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title("Seam Overlay (Red)")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ResUNet Weld Seam Pipeline")
    parser.add_argument("--image", required=True, help="Path to test ROI image (cropped workpiece)")
    parser.add_argument("--model", default="best_resunet_seam.pth", help="Path to best_resunet_seam.pth weights")
    args = parser.parse_args()
    
    test_image(args.image, args.model)
