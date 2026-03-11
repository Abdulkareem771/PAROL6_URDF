import os
import random
import shutil

# ===== CONFIG =====
DATASET_DIR = "dataset"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# ===== PATHS =====
images_dir = os.path.join(DATASET_DIR, "images")
labels_dir = os.path.join(DATASET_DIR, "labels")

splits = ["train", "val", "test"]

for split in splits:
    os.makedirs(os.path.join(images_dir, split), exist_ok=True)
    os.makedirs(os.path.join(labels_dir, split), exist_ok=True)

# ===== COLLECT IMAGES =====
images = [
    f for f in os.listdir(images_dir)
    if f.lower().endswith(IMAGE_EXTENSIONS)
]

random.shuffle(images)

total = len(images)
train_end = int(total * TRAIN_RATIO)
val_end = train_end + int(total * VAL_RATIO)

train_images = images[:train_end]
val_images = images[train_end:val_end]
test_images = images[val_end:]

# ===== MOVE FILES =====
def move_files(image_list, split):
    for img in image_list:
        src_img = os.path.join(images_dir, img)
        dst_img = os.path.join(images_dir, split, img)

        label = os.path.splitext(img)[0] + ".txt"
        src_label = os.path.join(labels_dir, label)
        dst_label = os.path.join(labels_dir, split, label)

        if os.path.exists(src_label):
            shutil.move(src_img, dst_img)
            shutil.move(src_label, dst_label)
        else:
            print(f"⚠️ Label missing for {img}, skipping.")

move_files(train_images, "train")
move_files(val_images, "val")
move_files(test_images, "test")

print("✅ Dataset split completed!")
print(f"Train: {len(train_images)} images")
print(f"Val:   {len(val_images)} images")
print(f"Test:  {len(test_images)} images")