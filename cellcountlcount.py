import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def count_cells_otsu(image_np, min_area=25, max_area=600):
    """
    Count cells using Otsu's thresholding method.
    Assumes a grayscale image.
    """
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image_np, (5, 5), 0)
    # Otsu thresholding with inversion (cells appear white on black)
    ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filter contours by area to avoid counting noise
    cell_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]
    count = len(cell_contours)
    return count, cell_contours, thresh

def count_cells_adaptive(image_np, min_area=25, max_area=600):
    """
    Count cells using adaptive thresholding.
    Assumes a grayscale image.
    """
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image_np, (5, 5), 0)
    # Adaptive thresholding (mean method) with inversion
    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                            cv2.THRESH_BINARY_INV, 11, 3)
    # Find contours in the adaptive thresholded image
    contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filter contours by area to avoid counting noise
    cell_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]
    count = len(cell_contours)
    return count, cell_contours, adaptive_thresh

def display_results(image_np, contours, method_name, count, thresh_img):
    """
    Display the original image with detected cell contours and the thresholded image.
    """
    # Draw contours on a color version of the original image
    image_with_contours = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)
    
    plt.figure(figsize=(12, 6))
    plt.suptitle(f"Method: {method_name} - Detected Cells: {count}", fontsize=16)
    
    plt.subplot(1, 2, 1)
    plt.imshow(image_with_contours)
    plt.title("Detected Cells")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(thresh_img, cmap='gray')
    plt.title("Thresholded Image")
    plt.axis("off")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def process_images(image_paths):
    """
    Process a list of images and count cells using two methods.
    Returns a dictionary with counts for each image.
    """
    results = {}
    for image_path in image_paths:
        # Open image using PIL and convert to grayscale (as numpy array)
        image = Image.open(image_path).convert("L")
        image_np = np.array(image)
        print(f"\nProcessing {image_path}...")
        
        # Method 1: Otsu's thresholding
        count_otsu, contours_otsu, thresh_otsu = count_cells_otsu(image_np)
        # Method 2: Adaptive thresholding
        count_adaptive, contours_adaptive, thresh_adaptive = count_cells_adaptive(image_np)
        
        print(f"Method 1 (Otsu): {count_otsu} cells")
        print(f"Method 2 (Adaptive): {count_adaptive} cells")
        
        results[image_path] = {
            "otsu": count_otsu,
            "adaptive": count_adaptive
        }
        
        # Optionally, display the results for each method
        display_results(image_np, contours_otsu, "Otsu Thresholding", count_otsu, thresh_otsu)
        display_results(image_np, contours_adaptive, "Adaptive Thresholding", count_adaptive, thresh_adaptive)
    
    return results

if __name__ == "__main__":
    # List of images to process (ensure these files are in your working directory)
    image_paths = ["1.tif", "2.tif", "3.tif", "4.tif", "5.tif"]
    
    # Process images and get cell counts
    results = process_images(image_paths)
    
    # Print a summary of results
    print("\nSummary of cell counts per image:")
    for img_path, counts in results.items():
        print(f"{img_path}: Otsu = {counts['otsu']} cells, Adaptive = {counts['adaptive']} cells")
