#!/usr/bin/env python3
"""
Interactive Image Cropper
Allows you to crop images from a source folder and save them to a destination folder.
"""

import cv2
import os
from pathlib import Path

current_dir = Path(__file__)
project_dir = current_dir.parent

# Path to the folder containing images to be cropped
SOURCE_FOLDER = project_dir / "data" / "raw_model1_data"    # replace with your folder path

# Path to the folder containing the cropped images
DEST_FOLDER = project_dir / "data" / "cropped_model1_data"    # replace with your folder path

class ImageCropper:
    def __init__(self, source_folder=SOURCE_FOLDER, dest_folder=DEST_FOLDER):
        """
        Initialize the Image Cropper.
        
        Args:
            source_folder (str): Folder containing raw images
            dest_folder (str): Folder to save cropped images
        """
        self.source_folder = Path(source_folder)
        self.dest_folder = Path(dest_folder)
        
        # Create destination folder if it doesn't exist
        self.dest_folder.mkdir(parents=True, exist_ok=True)
        
        # Supported image extensions
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
    def get_image_files(self):
        """Get all image files from the source folder."""
        image_files = []
        
        if not self.source_folder.exists():
            print(f"Error: Source folder '{self.source_folder}' does not exist!")
            return image_files
        
        for file_path in self.source_folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.image_extensions:
                image_files.append(file_path)
        
        return sorted(image_files)
    
    def crop_image(self, image_path):
        """
        Display image and allow user to select crop region.
        
        Args:
            image_path (Path): Path to the image file
            
        Returns:
            bool: True if image was cropped and saved, False otherwise
        """
        # Load the image
        image = cv2.imread(str(image_path))
        
        if image is None:
            print(f"Error: Could not load image '{image_path}'")
            return False
        
        # Create a copy for display
        display_image = image.copy()
        
        # Window name for consistent reference
        window_name = "Select Crop Region - Press SPACE/ENTER to confirm, C to cancel, ESC to quit"
        
        print(f"\nProcessing: {image_path.name}")
        print("Instructions:")
        print("  - Click and drag to select the crop region")
        print("  - Press SPACE or ENTER to confirm the crop")
        print("  - Press 'c' to cancel and skip this image")
        print("  - Press ESC to exit the program")
        print("  - You can resize, maximize, or minimize the window as needed")
        print("  - Double-click the title bar to maximize/restore")
        
        # Create a named window that is resizable (allows maximize/minimize)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Select ROI (Region of Interest)
        # This opens an interactive window where you can draw a rectangle
        roi = cv2.selectROI(
            window_name,
            display_image,
            showCrosshair=True,
            fromCenter=False
        )
        
        # Close the selection window
        cv2.destroyAllWindows()
        
        # Check if a valid ROI was selected
        x, y, w, h = roi
        
        if w == 0 or h == 0:
            print("  ⚠ No region selected. Skipping this image.")
            return False
        
        # Crop the image
        cropped_image = image[y:y+h, x:x+w]
        
        # Generate output filename
        output_path = self.dest_folder / image_path.name
        
        # If file already exists, add a number suffix
        counter = 1
        while output_path.exists():
            stem = image_path.stem
            suffix = image_path.suffix
            output_path = self.dest_folder / f"{stem}_crop_{counter}{suffix}"
            counter += 1
        
        # Save the cropped image
        cv2.imwrite(str(output_path), cropped_image)
        print(f"  ✓ Cropped image saved to: {output_path}")
        
        return True
    
    def process_all_images(self):
        """Process all images in the source folder."""
        image_files = self.get_image_files()
        
        if not image_files:
            print(f"No images found in '{self.source_folder}'")
            return
        
        print(f"Found {len(image_files)} image(s) in '{self.source_folder}'")
        print(f"Cropped images will be saved to '{self.dest_folder}'")
        print("=" * 60)
        
        processed_count = 0
        skipped_count = 0
        
        for image_path in image_files:
            result = self.crop_image(image_path)
            
            if result:
                processed_count += 1
            else:
                skipped_count += 1
        
        print("\n" + "=" * 60)
        print(f"Processing complete!")
        print(f"  Cropped: {processed_count}")
        print(f"  Skipped: {skipped_count}")
        print(f"  Total: {len(image_files)}")


def main():
    """Main function to run the image cropper."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Interactive Image Cropper - Crop images and save to a new folder"
    )
    parser.add_argument(
        "--source",
        "-s",
        default=SOURCE_FOLDER,
        help="Source folder containing images (default: raw_images)"
    )
    parser.add_argument(
        "--dest",
        "-d",
        default=DEST_FOLDER,
        help="Destination folder for cropped images (default: cropped_images)"
    )
    
    args = parser.parse_args()
    
    # Create and run the cropper
    cropper = ImageCropper(source_folder=args.source, dest_folder=args.dest)
    cropper.process_all_images()


if __name__ == "__main__":
    main()
