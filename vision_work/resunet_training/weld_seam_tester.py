import argparse
import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# ── Residual Block ───────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
        )
        self.skip = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.skip = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.skip(x))


# ── Encoder (Down-sampling) ──────────────────────────────────────────────────
class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.res  = ResBlock(in_c, out_c)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.res(x)
        x    = self.pool(skip)
        return skip, x


# ── Bridge ───────────────────────────────────────────────────────────────────
class Bridge(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.res = ResBlock(in_c, out_c)

    def forward(self, x):
        return self.res(x)


# ── Decoder (Up-sampling) ────────────────────────────────────────────────────
class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up  = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        self.res = ResBlock(out_c * 2, out_c)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.res(x)


# ── Full ResUNet ─────────────────────────────────────────────────────────────
class ResUNet(nn.Module):
    def __init__(self, in_c=3, out_c=1):
        super().__init__()
        self.e1     = EncoderBlock(in_c,   64)
        self.e2     = EncoderBlock(64,    128)
        self.e3     = EncoderBlock(128,   256)
        self.e4     = EncoderBlock(256,   512)
        self.bridge = Bridge(512, 1024)
        self.d4     = DecoderBlock(1024,  512)
        self.d3     = DecoderBlock(512,   256)
        self.d2     = DecoderBlock(256,   128)
        self.d1     = DecoderBlock(128,    64)
        self.head   = nn.Conv2d(64, out_c, kernel_size=1)

    def forward(self, x):
        s1, x = self.e1(x)
        s2, x = self.e2(x)
        s3, x = self.e3(x)
        s4, x = self.e4(x)
        x = self.bridge(x)
        x = self.d4(x, s4)
        x = self.d3(x, s3)
        x = self.d2(x, s2)
        x = self.d1(x, s1)
        return self.head(x)

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

    # Save composite results plot
    os.makedirs("test_results", exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(mask_resized, cmap='gray')
    axes[1].set_title("Predicted Seam Mask")
    axes[1].axis("off")

    if skeleton_overlay is not None:
        axes[2].imshow(cv2.cvtColor(skeleton_overlay, cv2.COLOR_BGR2RGB))
        axes[2].set_title("Skeleton Centerline (Green)")
    else:
        axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[2].set_title("Seam Overlay (Red)")
    axes[2].axis("off")

    plt.tight_layout()
    results_path = f"test_results/{base_name}_results.png"
    plt.savefig(results_path, dpi=150, bbox_inches="tight")
    print(f"Saved combined results to {results_path}")

    try:
        plt.show()
    except Exception:
        pass  # No display available (headless mode) — results already saved to file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ResUNet Weld Seam Pipeline")
    parser.add_argument("--image", required=True, help="Path to test ROI image (cropped workpiece)")
    parser.add_argument("--model", default="best_resunet_seam.pth", help="Path to best_resunet_seam.pth weights")
    args = parser.parse_args()

    test_image(args.image, args.model)
