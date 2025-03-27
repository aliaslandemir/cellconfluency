import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# ---------------------------
# Global Calibration Parameters
# ---------------------------
# At 10x magnification with a sensor pixel size of 6.5 µm,
# the effective pixel size is about 6.5/10 = 0.65 µm per pixel.
PIXEL_SIZE = 0.65  # µm per pixel
# Each pixel covers an area of (PIXEL_SIZE)^2 in µm².
AREA_CONVERSION = PIXEL_SIZE**2  # ~0.4225 µm² per pixel

# Hemocytometer square volume: 1 mm² area × 0.1 mm depth = 0.1 mm³,
# and since 1 mL = 1000 mm³, each square corresponds to 0.0001 mL.
VOLUME_PER_IMAGE = 0.0001  # mL

# Output directory for all saved figures
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# Preprocessing Function
# ---------------------------
def preprocess_image(image_np):
    """
    Preprocess the image: apply Gaussian blur to reduce noise.
    """
    blurred = cv2.GaussianBlur(image_np, (5, 5), 0)
    return blurred

# ---------------------------
# Segmentation Method 2: Canny Edge Detection + Dilation + Morphology
# ---------------------------
def segment_cells_method2(image_np, min_area=50, max_area=5000):
    """
    Segment cells using Canny edge detection followed by dilation and morphological closing.
    
    Args:
        image_np: Grayscale image (numpy array).
        min_area: Minimum contour area (in pixels) to be considered a cell.
        max_area: Maximum contour area (in pixels) to be considered a cell.
        
    Returns:
        cell_count: Number of detected cells.
        cell_contours: List of cell contours.
        binary_image: The final binary segmentation image.
        cell_areas: List of cell areas (converted to µm²).
    """
    preprocessed = preprocess_image(image_np)
    
    # Canny edge detection
    edges = cv2.Canny(preprocessed, threshold1=50, threshold2=150)
    
    # Dilate edges to connect cell boundaries
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Apply morphological closing to fill gaps
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Threshold to obtain a binary image
    ret, binary = cv2.threshold(closed, 50, 255, cv2.THRESH_BINARY)
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on area to remove noise
    cell_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]
    cell_count = len(cell_contours)
    
    # Convert each contour area from pixels to µm²
    cell_areas = [cv2.contourArea(cnt) * AREA_CONVERSION for cnt in cell_contours]
    
    return cell_count, cell_contours, binary, cell_areas

# ---------------------------
# Visualization Function
# ---------------------------
def display_segmentation(image_np, contours, binary_image, method_name, cell_count, image_title, save_path):
    """
    Display the original image overlaid with detected contours and the binary segmentation.
    Save the resulting figure to a file.
    """
    # Create an image copy with contours overlaid
    image_with_contours = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)
    
    plt.figure(figsize=(12, 6))
    plt.suptitle(f"{image_title} - {method_name} (Cells Detected: {cell_count})", fontsize=16)
    
    plt.subplot(1, 2, 1)
    plt.imshow(image_with_contours)
    plt.title("Overlay of Detected Cells")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(binary_image, cmap="gray")
    plt.title("Binary Segmentation")
    plt.axis("off")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()

# ---------------------------
# Process a Single Image File
# ---------------------------
def process_image_file(image_path, display_results=True):
    """
    Process one image file using segmentation method 2, count cells, and estimate concentration.
    
    Args:
        image_path: Path to the image file.
        display_results: If True, show and save the segmentation result.
        
    Returns:
        count: Number of cells detected.
        concentration: Estimated cells per mL.
    """
    # Load image and convert to grayscale using PIL
    image = Image.open(image_path).convert("L")
    image_np = np.array(image)
    
    # Segment cells using Method 2
    count, contours, binary, cell_areas = segment_cells_method2(image_np)
    
    # Calculate cell concentration (cells per mL)
    # Each image represents a hemocytometer square (volume = 0.0001 mL)
    concentration = count * 1e4
    
    # Save segmentation result image
    base_name = os.path.basename(image_path)
    save_filename = os.path.join(OUTPUT_DIR, f"segmentation_{base_name}_method2.png")
    if display_results:
        display_segmentation(image_np, contours, binary, "Method 2: Canny+Morphology", count,
                             image_title=base_name, save_path=save_filename)
    
    return count, concentration

# ---------------------------
# Main Script: Process Multiple Images and Save Summary Plots
# ---------------------------
if __name__ == "__main__":
    # List of image filenames (adjust paths if necessary)
    image_files = ["1.tif", "2.tif", "3.tif", "4.tif", "5.tif"]
    
    cell_counts = []
    concentrations = []
    
    # Process each image one by one
    for img_file in image_files:
        print(f"Processing {img_file}...")
        count, conc = process_image_file(img_file, display_results=True)
        print(f"  -> Detected {count} cells; Estimated Concentration: {conc:.0f} cells/mL")
        cell_counts.append(count)
        concentrations.append(conc)
    
    # Calculate overall statistics
    mean_conc = np.mean(concentrations)
    std_conc = np.std(concentrations)
    
    # ---------------------------
    # Create Summary Figures
    # ---------------------------
    # Figure 1: Boxplot of cells/mL across images
    plt.figure(figsize=(8, 6))
    plt.boxplot(concentrations, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    plt.title("Boxplot of Cells/mL (per Hemocytometer Square)")
    plt.ylabel("Cells/mL")
    plt.xticks([1], ["All Images"])
    boxplot_save_path = os.path.join(OUTPUT_DIR, "summary_boxplot_cells_per_ml.png")
    plt.savefig(boxplot_save_path)
    plt.close()
    
    # Figure 2: Bar plot with error bars showing individual image concentrations and overall average
    plt.figure(figsize=(10, 6))
    x = np.arange(len(concentrations))
    plt.bar(x, concentrations, color='lightgreen', alpha=0.8, label="Individual")
    plt.errorbar(x, concentrations, yerr=std_conc, fmt='none', ecolor='black', capsize=5)
    plt.axhline(mean_conc, color='red', linestyle='--', label=f"Mean = {mean_conc:.0f}")
    plt.title("Cells/mL per Image")
    plt.xlabel("Image Index")
    plt.ylabel("Cells/mL")
    plt.xticks(x, [f"Img {i+1}" for i in x])
    plt.legend()
    barplot_save_path = os.path.join(OUTPUT_DIR, "summary_barplot_cells_per_ml.png")
    plt.savefig(barplot_save_path)
    plt.close()
    
    # Print summary
    print("\nSummary of Cell Concentrations (cells/mL):")
    for i, conc in enumerate(concentrations):
        print(f"Image {i+1}: {conc:.0f} cells/mL")
    print(f"\nOverall Mean: {mean_conc:.0f} cells/mL, Standard Deviation: {std_conc:.0f} cells/mL")
    
    # Optionally, save the summary to a text file
    summary_text = "Summary of Cell Concentrations (cells/mL):\n"
    for i, conc in enumerate(concentrations):
        summary_text += f"Image {i+1}: {conc:.0f} cells/mL\n"
    summary_text += f"\nOverall Mean: {mean_conc:.0f} cells/mL, Standard Deviation: {std_conc:.0f} cells/mL\n"
    
    with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w") as f:
        f.write(summary_text)
