import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import measure, morphology, segmentation, color, feature
from scipy import ndimage

# ------------------------------
# Configuration
# ------------------------------
PIXELS_PER_UM = 2.21
INPUT_DIR = "images/"
OUTPUT_CSV = "outputs/cell_analysis_results.csv"
OUTPUT_SEGMENTED_DIR = "outputs"  # Folder for saving segmentation images
MIN_CELL_AREA_UM2 = 50  # Minimum cell area in µm²

# ------------------------------
# Create output folder if it doesn't exist
# ------------------------------
os.makedirs(OUTPUT_SEGMENTED_DIR, exist_ok=True)

def process_image(img_path):
    """
    Read and preprocess a potentially 16-bit brightfield-like image,
    then perform segmentation optimized for elongated adherent cells.
    Returns the color (RGB) image, label matrix of segmented regions,
    and the contrast-enhanced grayscale image used for intensity measurement.
    """
    # 1. Load and handle 16-bit
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")

    # Convert 16-bit images to 8-bit for processing
    if img.dtype == np.uint16:
        img = cv2.convertScaleAbs(img, alpha=(255.0 / 65535.0))

    # 2. Convert to grayscale and a separate RGB version
    if len(img.shape) == 3 and img.shape[2] >= 3:
        original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        # Already single-channel
        original = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        gray = img

    # 3. Preprocessing: 
    # Denoise + optional morphological top-hat to highlight elongated cells
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)

    # (Optional) Top-hat transform to isolate bright/dark features
    # kernel_th = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    # tophat = cv2.morphologyEx(denoised, cv2.MORPH_TOPHAT, kernel_th)
    # You can experiment with top-hat or bottom-hat depending on your images:
    # gray_prepped = tophat
    # For now, let's continue using denoised directly:

    # 4. Contrast enhancement with CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # 5. Invert if cells appear dark on bright background 
    # (Brightfield images often have dark cells). Uncomment if needed.
    # enhanced = cv2.bitwise_not(enhanced)

    # 6. Thresholding 
    # Using adaptive threshold for uneven illumination
    thresh = cv2.adaptiveThreshold(enhanced, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 5)
    # For elongated cells, sometimes a slightly smaller blockSize or 
    # different constant can help. Adjust as needed.

    # 7. Morphological operations
    # Use elongated kernel to preserve elongated structures
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 7))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 8. Distance transform + local peak detection
    distance = ndimage.distance_transform_edt(closed)
    coords = feature.peak_local_max(distance, footprint=np.ones((15, 15)), labels=closed)
    local_maxi = np.zeros(distance.shape, dtype=bool)
    if coords.size:
        local_maxi[tuple(coords.T)] = True

    # 9. Watershed
    markers = measure.label(local_maxi)
    ws_labels = segmentation.watershed(-distance, markers, mask=closed)

    # 10. Remove small objects
    min_area_pixels = MIN_CELL_AREA_UM2 * (PIXELS_PER_UM ** 2)
    filtered_labels = morphology.remove_small_objects(ws_labels, min_size=min_area_pixels)

    return original, filtered_labels, enhanced

def analyze_cells(label_image, intensity_image):
    """
    Extract region properties for each cell, including shape and intensity features.
    """
    props = measure.regionprops(label_image, intensity_image=intensity_image)
    results = []

    # Calculate total cell area to get confluency later
    total_area_px = 0

    for prop in props:
        # Basic geometry
        area_px = prop.area
        area_um2 = area_px / (PIXELS_PER_UM ** 2)
        perimeter_um = prop.perimeter / PIXELS_PER_UM

        # Additional shape descriptors
        convex_hull_perimeter = measure.perimeter(prop.convex_image)
        convexity = (convex_hull_perimeter / prop.perimeter) if prop.perimeter > 0 else 0
        solidity = prop.solidity
        circularity = (4 * np.pi * prop.area) / (prop.perimeter ** 2) if prop.perimeter > 0 else 0
        eccentricity = prop.eccentricity

        # Intensity features
        intensity_mean = prop.mean_intensity
        coords = prop.coords
        intensity_std = np.std(intensity_image[coords[:, 0], coords[:, 1]])

        # Accumulate total area
        total_area_px += area_px

        results.append({
            "Area_um2": area_um2,
            "Perimeter_um": perimeter_um,
            "Convexity": convexity,
            "Solidity": solidity,
            "Circularity": circularity,
            "Eccentricity": eccentricity,
            "Intensity_Mean": intensity_mean,
            "Intensity_Std": intensity_std
        })

    return pd.DataFrame(results), total_area_px

def visualize_and_save(original, label_image, perimeter_map_path, colored_path, df):
    """
    - Overlays cell contours and perimeter values (in µm) on 'original'
    - Saves a color-labeled segmentation image
    - perimeter_map_path: file path to save perimeter-annotated image
    - colored_path: file path to save color-labeled segmentation image
    """
    # Create the colored label image
    colored_labels = color.label2rgb(label_image, image=original, bg_label=0, alpha=0.3)

    # -------------------------------------
    # 1) Annotated image with perimeter
    # -------------------------------------
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    ax1.imshow(original)
    ax1.set_title("Cells with Perimeter (µm)")

    # We need regionprops again to get centroids + perimeter to annotate
    props = measure.regionprops(label_image)
    for prop in props:
        perimeter_um = (prop.perimeter / PIXELS_PER_UM)
        contour = measure.find_contours(prop.image, 0.5)
        if len(contour) > 0:
            c = contour[0]
            y0, x0, _, _ = prop.bbox
            c[:, 0] += y0
            c[:, 1] += x0
            ax1.plot(c[:, 1], c[:, 0], linewidth=1, color='yellow')
            # Place perimeter text near centroid
            cy, cx = prop.centroid
            ax1.text(cx, cy, f"{perimeter_um:.1f}",
                     color='red', fontsize=8, ha='center', va='center')

    ax1.axis('off')
    fig1.tight_layout()
    fig1.savefig(perimeter_map_path, dpi=150)
    plt.close(fig1)

    # -------------------------------------
    # 2) Color-labeled segmentation image
    # -------------------------------------
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    ax2.imshow(colored_labels)
    ax2.set_title("Color-Labeled Segmentation")
    ax2.axis('off')
    fig2.tight_layout()
    fig2.savefig(colored_path, dpi=150)
    plt.close(fig2)

def main():
    all_results = []
    summary_list = []  # to store confluency and cell count for each image

    for file in os.listdir(INPUT_DIR):
        if file.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg')):
            img_path = os.path.join(INPUT_DIR, file)
            try:
                original, labels, enhanced = process_image(img_path)
            except Exception as e:
                print(f"Error processing {file}: {e}")
                continue

            df, total_area_px = analyze_cells(labels, enhanced)

            # Add image filename to each row
            df['Image'] = file
            all_results.append(df)

            # ---------------------------------------
            # Calculate confluency & cell count
            # ---------------------------------------
            # Confluency = fraction of image area occupied by cells
            total_pixels = labels.shape[0] * labels.shape[1]
            confluency = (total_area_px / total_pixels) * 100.0
            cell_count = len(df)  # each region ~ one cell

            summary_list.append({
                "Image": file,
                "Confluency(%)": confluency,
                "Estimated_Cells": cell_count
            })

            # ---------------------------------------
            # Save segmentation images
            # ---------------------------------------
            base_name = os.path.splitext(file)[0]
            perimeter_map_path = os.path.join(
                OUTPUT_SEGMENTED_DIR, f"{base_name}_perimeter.png"
            )
            colored_path = os.path.join(
                OUTPUT_SEGMENTED_DIR, f"{base_name}_colored.png"
            )
            visualize_and_save(original, labels, perimeter_map_path, colored_path, df)

    # -------------------
    # Combine and save results
    # -------------------
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv(OUTPUT_CSV, index=False)
        print("Analysis Summary (All Regions Across All Images):")
        print(final_df.describe())

        # Save summary of confluency + cell counts
        summary_df = pd.DataFrame(summary_list)
        summary_df.to_csv("outputs/image_summaries.csv", index=False)
        print("\nImage-Level Summary (Confluency, Cell Count):")
        print(summary_df)

        # -------------------
        # One combined plot for all images
        # -------------------
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        # Histograms for entire dataset
        final_df.hist(column='Area_um2', bins=30, ax=axes[0, 0])
        axes[0, 0].set_title("Area (µm²) - All Images")
        final_df.hist(column='Perimeter_um', bins=30, ax=axes[0, 1])
        axes[0, 1].set_title("Perimeter (µm) - All Images")
        final_df.hist(column='Convexity', bins=30, ax=axes[0, 2])
        axes[0, 2].set_title("Convexity - All Images")
        final_df.hist(column='Circularity', bins=30, ax=axes[1, 0])
        axes[1, 0].set_title("Circularity - All Images")
        final_df.hist(column='Solidity', bins=30, ax=axes[1, 1])
        axes[1, 1].set_title("Solidity - All Images")
        final_df.hist(column='Eccentricity', bins=30, ax=axes[1, 2])
        axes[1, 2].set_title("Eccentricity - All Images")

        plt.tight_layout()
        plt.savefig("outputs/combined_feature_distributions.png", dpi=150)
        plt.close()
        print("Combined feature distribution plot saved as 'combined_feature_distributions.png'.")

    else:
        print("No images processed.")

if __name__ == "__main__":
    main()
