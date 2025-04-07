import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Directory to save exported figures
EXPORT_DIR = "figures"
if not os.path.exists(EXPORT_DIR):
    os.makedirs(EXPORT_DIR)

def add_scale_bar(image, pixel_to_micron, scale_bar_length_um=200, bar_color=(255, 0, 0), thickness=5, margin=20):
    """
    Overlays a scale bar on the BGR image.
    """
    scale_bar_length_px = int(scale_bar_length_um / pixel_to_micron)
    height, width, _ = image.shape
    x_start = margin
    y_start = height - margin
    x_end = x_start + scale_bar_length_px

    cv2.line(image, (x_start, y_start), (x_end, y_start), bar_color, thickness)
    text = f"{scale_bar_length_um} um"  # ASCII-safe label
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, (x_start, y_start - 10), font, 1, bar_color, 2, cv2.LINE_AA)
    return image

def count_cells_adaptive(image_np, min_area=50, max_area=600):
    """
    Counts cells via adaptive thresholding on a grayscale image.
    
    Returns:
      count: Number of detected cells.
      cell_contours: List of contours that pass the area filter.
      thresh_img: The binary thresholded image.
    """
    blurred = cv2.GaussianBlur(image_np, (7, 7), 0)
    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                            cv2.THRESH_BINARY_INV, 9, 3)
    contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cell_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]
    count = len(cell_contours)
    return count, cell_contours, adaptive_thresh

def display_results(image_np, contours, method_name, count, thresh_img, pixel_to_micron, scale_bar_length_um, export_filename=None):
    """
    Displays and exports a figure with detected cell contours (with scale bar) and the thresholded image.
    """
    image_with_contours = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)
    image_with_contours = add_scale_bar(image_with_contours, pixel_to_micron, scale_bar_length_um)
    image_rgb = cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(12, 6))
    plt.suptitle(f"{method_name}: Detected Cells = {count}", fontsize=16)
    
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title("Detected Cells with Scale Bar")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(thresh_img, cmap='gray')
    plt.title("Thresholded Image")
    plt.axis("off")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if export_filename:
        plt.savefig(os.path.join(EXPORT_DIR, export_filename), dpi=300)
    plt.show()
    plt.close()

def compute_cells_per_ml_hemacytometer(cell_count, squares_count, dilution_factor=2):
    """
    Computes total cells per mL using the classical hemocytometer formula:
    
      Total cells/mL = (cell_count * dilution_factor * 10,000) / squares_count
    """
    return (cell_count * dilution_factor * 10000) / squares_count

def plot_cells_per_ml_stats(cells_per_ml, image_labels, title="Cells per mL per Image", export_filename=None):
    """
    Creates and exports a bar chart showing cells/mL for each image.
    Also annotates each bar and plots a horizontal line for the average.
    """
    plt.style.use('ggplot')  # or 'classic', 'bmh', 'dark_background', etc.
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(cells_per_ml))
    bars = ax.bar(x, cells_per_ml, color='skyblue', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(image_labels, fontsize=12)
    ax.set_xlabel("Image", fontsize=14)
    ax.set_ylabel("Cells per mL", fontsize=14)
    ax.set_title(title, fontsize=16)
    
    # Annotate each bar with its value
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12)
    
    # Average line annotation
    avg = np.mean(cells_per_ml)
    ax.axhline(avg, color='red', linestyle='--', linewidth=2, label=f'Average = {avg:.0f}')
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    if export_filename:
        plt.savefig(os.path.join(EXPORT_DIR, export_filename), dpi=300)
    plt.show()
    plt.close()

def process_images(image_paths, squares_count=1, dilution_factor=2, 
                   min_area=50, max_area=600, pixel_to_micron=0.63, scale_bar_length_um=200):
    """
    Processes a list of images to:
      - Read the image in grayscale.
      - Compute physical dimensions using pixel_to_micron.
      - Count cells using adaptive thresholding.
      - Scale raw counts to a standard 1 mm² area.
      - Compute cells/mL via the classical hemocytometer formula.
      - Display and export intermediate figures.
      
    Returns:
      dict: Mapping from image file to result details.
    """
    results = {}
    cells_per_ml_list = []
    image_labels = []
    
    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
        
        # Read image and compute physical dimensions
        image = Image.open(image_path).convert("L")
        image_np = np.array(image)
        height, width = image_np.shape
        phys_width_mm = (width * pixel_to_micron) / 1000
        phys_height_mm = (height * pixel_to_micron) / 1000
        area_mm2 = phys_width_mm * phys_height_mm
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        image_labels.append(base_name)
        
        print(f"\nProcessing {image_path}: {width}x{height} px, ~{phys_width_mm:.3f}x{phys_height_mm:.3f} mm (Area {area_mm2:.3f} mm²)")
        
        # Count cells using adaptive thresholding
        count, contours, thresh_img = count_cells_adaptive(image_np, min_area, max_area)
        scaled_count = count * (1 / area_mm2)  # Scale to 1 mm²
        concentration = compute_cells_per_ml_hemacytometer(scaled_count, squares_count, dilution_factor)
        
        results[image_path] = {"raw_cell_count": count,
                               "scaled_cell_count": scaled_count,
                               "cells_per_mL": concentration}
        cells_per_ml_list.append(concentration)
        
        print(f"Adaptive thresholding: {count} raw cells, {scaled_count:.1f} cells in 1 mm², {concentration:.1f} cells/mL")
        export_filename = f"{base_name}_results.png"
        display_results(image_np, contours, "Adaptive Thresholding", count, thresh_img,
                        pixel_to_micron, scale_bar_length_um, export_filename)
    
    summary_export = "cells_per_ml_summary.png"
    plot_cells_per_ml_stats(cells_per_ml_list, image_labels, title="Cells per mL per Image", export_filename=summary_export)
    
    return results

if __name__ == "__main__":
    # List of image files (ensure these files are in your working directory)
    image_paths = ["1.tif", "2.tif", "3.tif", "4.tif", "5.tif"]
    results = process_images(image_paths, squares_count=1, dilution_factor=2,
                             min_area=50, max_area=600, pixel_to_micron=0.63, scale_bar_length_um=200)
    
    print("\nSummary:")
    for img, res in results.items():
        print(f"{img}: raw count = {res['raw_cell_count']} cells, "
              f"scaled count = {res['scaled_cell_count']:.1f} cells in 1 mm², "
              f"{res['cells_per_mL']:.1f} cells/mL")
