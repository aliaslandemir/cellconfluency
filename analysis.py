import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import measure, morphology, segmentation, color, feature, filters
from scipy import ndimage
from sklearn.mixture import GaussianMixture

# ------------------------------
# Configuration
# ------------------------------
PIXELS_PER_UM = 2.21

INPUT_DIR = "images/"

# Method1:
OUTPUT_CSV_METHOD1 = "outputs/cell_analysis_results_method1.csv"
OUTPUT_SEGMENTED_DIR_METHOD1 = "outputs"

# Method2:
OUTPUT_CSV_METHOD2 = "outputs_method2/cell_analysis_results_method2.csv"
OUTPUT_SEGMENTED_DIR_METHOD2 = "outputs_method2"

# Method3_
OUTPUT_CSV_METHOD3 = "outputs_method3/cell_analysis_results_method3.csv"
OUTPUT_SEGMENTED_DIR_METHOD3 = "outputs_method3"

# Minimum cell area (µm²)
MIN_CELL_AREA_UM2 = 100

# Create output folders if they don't exist
os.makedirs(OUTPUT_SEGMENTED_DIR_METHOD1, exist_ok=True)
os.makedirs(OUTPUT_SEGMENTED_DIR_METHOD2, exist_ok=True)
os.makedirs(OUTPUT_SEGMENTED_DIR_METHOD3, exist_ok=True)

# ------------------------------
# Shared Functions
# ------------------------------
def convert_to_8bit_if_needed(img):
    """Convert 16-bit to 8-bit if necessary."""
    if img.dtype == np.uint16:
        img = cv2.convertScaleAbs(img, alpha=(255.0 / 65535.0))
    return img

def analyze_cells(label_image, intensity_image):
    """
    Extract region properties for each cell, including shape and intensity.
    Returns a DataFrame with cell properties and the total area in pixels.
    """
    props = measure.regionprops(label_image, intensity_image=intensity_image)
    results = []
    total_area_px = 0

    for prop in props:
        area_px = prop.area
        area_um2 = area_px / (PIXELS_PER_UM ** 2)
        perimeter_um = prop.perimeter / PIXELS_PER_UM

        # shape descriptors
        convex_hull_perimeter = measure.perimeter(prop.convex_image)
        convexity = (convex_hull_perimeter / prop.perimeter) if prop.perimeter > 0 else 0
        solidity = prop.solidity
        circularity = (4 * np.pi * prop.area) / (prop.perimeter ** 2) if prop.perimeter > 0 else 0
        eccentricity = prop.eccentricity

        # intensity
        intensity_mean = prop.mean_intensity
        coords = prop.coords
        intensity_std = np.std(intensity_image[coords[:, 0], coords[:, 1]])

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

def compute_final_metrics(labels, df):
    """
    Compute confluency, estimate total cells on T-25, etc.
    Returns (confluency%, cell_count, cells_per_mL).
    """
    total_area_px = df["Area_um2"].sum() * (PIXELS_PER_UM ** 2)  # convert area back to pixels

    h, w = labels.shape
    total_pixels = h * w
    confluency = (total_area_px / total_pixels) * 100.0
    cell_count = len(df)

    # Culture assumptions
    CULTURE_AREA_CM2 = 25       # T-25 flask area
    RESUSPENSION_VOLUME_ML = 3

    # Imaged area
    imaged_area_um2 = (w / PIXELS_PER_UM) * (h / PIXELS_PER_UM)
    imaged_area_cm2 = imaged_area_um2 / 1e8  # 1 cm² = 1e8 µm²

    density_cells_per_cm2 = cell_count / imaged_area_cm2
    total_cells = density_cells_per_cm2 * CULTURE_AREA_CM2
    cells_per_mL = total_cells / RESUSPENSION_VOLUME_ML

    return confluency, cell_count, cells_per_mL

def visualize_and_save(original, labels, perimeter_map_path, colored_path, confluency_percent, cells_per_mL, method_name=""):
    """
    Overlays cell contours and perimeter on 'original' and saves a color-labeled segmentation image.
    """
    # Color-labeled image
    colored_labels = color.label2rgb(labels, image=original, bg_label=0, alpha=0.3)

    # 1) Annotated perimeter map
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    ax1.imshow(original)
    ax1.set_title(
        f"Cells with Perimeter (µm) [{method_name}] "
        f"Confluency: {confluency_percent:.1f}%; Cells/mL: {cells_per_mL:.0f}"
    )

    props = measure.regionprops(labels)
    for prop in props:
        perimeter_um = prop.perimeter / PIXELS_PER_UM
        contour = measure.find_contours(prop.image, 0.5)
        if len(contour) > 0:
            c = contour[0]
            y0, x0, _, _ = prop.bbox
            c[:, 0] += y0
            c[:, 1] += x0
            ax1.plot(c[:, 1], c[:, 0], linewidth=1, color='yellow')
            cy, cx = prop.centroid
            ax1.text(cx, cy, f"{perimeter_um:.1f}", color='red', fontsize=7, ha='center', va='center')

    ax1.axis('off')
    fig1.tight_layout()
    fig1.savefig(perimeter_map_path, dpi=150)
    plt.close(fig1)

    # 2) Colored segmentation
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    ax2.imshow(colored_labels)
    ax2.set_title(
        f"Color-Labeled Segmentation [{method_name}] "
        f"Confluency: {confluency_percent:.1f}%; Cells/mL: {cells_per_mL:.0f}"
    )
    ax2.axis('off')
    fig2.tight_layout()
    fig2.savefig(colored_path, dpi=150)
    plt.close(fig2)

# ------------------------------
# Method 1
# ------------------------------
def process_image_method1(img_path):
    """
    Original approach:
      - Bilateral denoise
      - CLAHE
      - Adaptive Threshold + morphological open/close
      - Distance transform + watershed
      - Remove small objects
    """
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")

    img = convert_to_8bit_if_needed(img)

    # Convert to grayscale & keep an RGB version
    if len(img.shape) == 3 and img.shape[2] >= 3:
        original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        original = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        gray = img

    # Bilateral
    denoised = cv2.bilateralFilter(gray, 11, 80, 80)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(7, 7))
    enhanced = clahe.apply(denoised)

    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 51, 7
    )

    # Morphological open/close
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 11))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Distance transform + watershed
    distance = ndimage.distance_transform_edt(closed)
    coords = feature.peak_local_max(distance, footprint=np.ones((60, 60)), labels=closed)
    local_maxi = np.zeros(distance.shape, dtype=bool)
    if coords.size:
        local_maxi[tuple(coords.T)] = True

    markers = measure.label(local_maxi)
    ws_labels = segmentation.watershed(-distance, markers, mask=closed)

    # Remove small objects
    min_area_pixels = MIN_CELL_AREA_UM2 * (PIXELS_PER_UM ** 2)
    filtered_labels = morphology.remove_small_objects(ws_labels, min_size=min_area_pixels)

    return original, filtered_labels, enhanced

# ------------------------------
# Method 2 (Top-Hat + Otsu + Watershed)
# ------------------------------
def process_image_method2(img_path):
    """
    - Morphological black-hat to enhance dark cells on a uniform gray background
    - Otsu threshold to binarize the enhanced image
    - Morphological open/close to clean small artifacts
    - Distance transform + watershed to separate touching cells
    - Remove small objects
    """
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")

    img = convert_to_8bit_if_needed(img)

    # Convert to grayscale & retain an RGB version for final visualization
    if len(img.shape) == 3 and img.shape[2] >= 3:
        original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        original = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        gray = img

    # Black-hat transform enhances dark objects on a lighter/gray background.
    kernel_blackhat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_blackhat)

    # Otsu threshold: since cells become brighter in the black-hat image, 
    # they will be segmented as foreground.
    otsu_val = filters.threshold_otsu(blackhat)
    binary = (blackhat > otsu_val).astype(np.uint8)

    # Clean up small artifacts with morphological operations.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Distance transform and watershed to separate touching cells.
    distance = ndimage.distance_transform_edt(closed)
    coords = feature.peak_local_max(distance, footprint=np.ones((30, 30)), labels=closed)
    local_maxi = np.zeros(distance.shape, dtype=bool)
    if coords.size:
        local_maxi[tuple(coords.T)] = True

    markers = measure.label(local_maxi)
    ws_labels = segmentation.watershed(-distance, markers, mask=closed)

    # Remove small objects (based on MIN_CELL_AREA_UM2)
    min_area_pixels = MIN_CELL_AREA_UM2 * (PIXELS_PER_UM ** 2)
    filtered_labels = morphology.remove_small_objects(ws_labels, min_size=min_area_pixels)

    # Return the original image, segmented labels, and the blackhat (used for thresholding)
    return original, filtered_labels, blackhat

# ------------------------------
# Method 3 (GMM Segmentation)
# ------------------------------
def process_image_method3(img_path):
    """
    - Flatten grayscale into 1D
    - Fit a 2-component Gaussian Mixture Model (background vs cells)
    - Reshape GMM labels into mask
    - Morphological cleanup + remove small objects
    """
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")

    img = convert_to_8bit_if_needed(img)

    # Convert to grayscale & keep RGB
    if len(img.shape) == 3 and img.shape[2] >= 3:
        original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        original = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        gray = img

    # Flatten for GMM
    flat = gray.reshape(-1, 1).astype(np.float32)

    # 2-component GMM
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
    gmm.fit(flat)

    # Predict cluster labels (0 or 1)
    labels_1d = gmm.predict(flat)
    mask = labels_1d.reshape(gray.shape)
    # Compare the mean intensities of the 2 clusters:
    cluster_means = gmm.means_.flatten()

    # We'll guess the cluster with the lower mean is "cell" if that suits your images:
    cell_cluster = np.argmin(cluster_means)
    binary = (mask == cell_cluster).astype(np.uint8)

    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Label
    labeled = measure.label(closed, connectivity=2)

    # Remove small objects
    min_area_pixels = MIN_CELL_AREA_UM2 * (PIXELS_PER_UM ** 2)
    filtered_labels = morphology.remove_small_objects(labeled, min_size=min_area_pixels)

    # We'll reuse the original grayscale as the intensity image
    return original, filtered_labels, gray

# ------------------------------
# Runner utility for each method
# ------------------------------
def run_method(process_func, output_seg_dir, output_csv, method_name):
    all_results = []
    summary_list = []

    for file in os.listdir(INPUT_DIR):
        if file.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg')):
            img_path = os.path.join(INPUT_DIR, file)
            try:
                original, labels, intensity_img = process_func(img_path)
            except Exception as e:
                print(f"Error processing {file} with {method_name}: {e}")
                continue

            df, _ = analyze_cells(labels, intensity_img)
            df['Image'] = file
            all_results.append(df)

            confluency, cell_count, cells_per_mL = compute_final_metrics(labels, df)
            summary_list.append({
                "Image": file,
                "Confluency(%)": confluency,
                "Estimated_Cells": cell_count
            })

            base_name = os.path.splitext(file)[0]
            perimeter_map_path = os.path.join(output_seg_dir, f"{base_name}_perimeter_{method_name}.png")
            colored_path = os.path.join(output_seg_dir, f"{base_name}_colored_{method_name}.png")

            visualize_and_save(
                original,
                labels,
                perimeter_map_path,
                colored_path,
                confluency,
                cells_per_mL,
                method_name=method_name
            )

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv(output_csv, index=False)
        print(f"[{method_name}] Analysis Summary:")
        print(final_df.describe())

        summary_df = pd.DataFrame(summary_list)
        summary_csv = os.path.join(os.path.dirname(output_csv), f"image_summaries_{method_name}.csv")
        summary_df.to_csv(summary_csv, index=False)
        print(f"\n[{method_name}] Image-Level Summary (Confluency, Cell Count):")
        print(summary_df)
    else:
        print(f"No images processed for {method_name}.")

# ------------------------------
# Main
# ------------------------------
def main():

    # Method1:
    run_method(process_image_method1, OUTPUT_SEGMENTED_DIR_METHOD1, OUTPUT_CSV_METHOD1, "Method1")

    # Method2: Top-Hat + Otsu + Watershed
    run_method(process_image_method2, OUTPUT_SEGMENTED_DIR_METHOD2, OUTPUT_CSV_METHOD2, "Method2")

    # Method3: GMM segmentation
    run_method(process_image_method3, OUTPUT_SEGMENTED_DIR_METHOD3, OUTPUT_CSV_METHOD3, "Method3")


if __name__ == "__main__":
    main()
