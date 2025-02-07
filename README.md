
```markdown
# Cell Confluency Analysis

Cell Confluency Analysis is a Python-based pipeline for the automated segmentation and analysis of retinal pigment epithelium (RPE) cells from bright-field absorbance images. The pipeline leverages classical image processing techniques—such as background subtraction, adaptive thresholding, edge detection with Laplacian of Gaussian, and marker-controlled watershed segmentation with h‑minima refinement—to detect and separate densely connected cells. In addition, it computes various cell features (e.g., area, perimeter, convexity, solidity, circularity, eccentricity, and intensity statistics) and provides annotated visualizations.

> **Note:** This project was developed for RPE cell images but may be adapted for other bright-field imaging applications with appropriate parameter tuning.

## Features

- **Automated Segmentation:**  
  - Background subtraction using a large-scale morphological opening.
  - Adaptive thresholding combined with a Laplacian of Gaussian (LoG) edge map.
  - Marker-controlled watershed segmentation with h‑minima transform for improved separation of connected cells.
  - Post-processing to remove small objects.

- **Cell Analysis:**  
  - Extraction of single-cell features (area, perimeter, convexity, solidity, circularity, eccentricity, intensity mean, and intensity standard deviation).
  - Calculation of overall confluency and cell counts.
  - Output results as CSV files.

- **Visualization:**  
  - Generation of annotated images with cell contours and perimeter annotations.
  - Creation of color-labeled segmentation overlays.
  - Combined feature distribution plots.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- Pandas
- Matplotlib
- scikit-image
- SciPy

You can install the required libraries using:

```bash
pip install -r requirements.txt
```

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/aliaslandemir/cellconfluency.git
   cd cellconfluency
   ```

2. **Set Up a Virtual Environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Usage

1. **Add Your Images:**  
   Place your bright-field absorbance images (16-bit or 8-bit) in the `images/` folder.

2. **Run the Analysis Script:**  
   From the project root directory, run:

   ```bash
   python analysis.py
   ```

   The script will:
   - Process all images in the `images/` folder.
   - Generate segmentation masks and extract cell features.
   - Save detailed cell features to `cell_analysis_results.csv`.
   - Save per-image summaries (confluency and cell counts) to `image_summaries.csv`.
   - Save annotated images and color-labeled segmentation images in the `segmented_outputs/` folder.
   - Generate a combined feature distribution plot (`combined_feature_distributions.png`).

## Pushing Changes to GitHub with VS Code

1. **Open Your Project in VS Code:**  
   Open the `cellconfluency` folder in VS Code.

2. **Open the Integrated Terminal:**  
   Press <kbd>Ctrl</kbd> + <kbd>`</kbd> (backtick) or select **Terminal > New Terminal**.

3. **Stage Your Changes:**

   ```bash
   git add .
   ```

4. **Commit Your Changes:**

   ```bash
   git commit -m "Your commit message"
   ```

5. **Push to GitHub:**

   ```bash
   git branch -M main
   git push -u origin main
   ```

   Alternatively, use the Source Control panel in VS Code to stage, commit, and push your changes.

## Project Structure

```
cellconfluency/
├── images/                  # Input bright-field images
├── outputs/       # Output folder for annotated segmentation images
├── analysis.py              # Main script for segmentation and analysis
├── requirements.txt         # Python package requirements
├── README.md                # This file
└── (additional files, if any)
```

## Contributing

Contributions are welcome! To contribute:
- Fork the repository.
- Create a new branch for your changes.
- Commit your changes with clear messages.
- Open a pull request for review.

## License

[Specify your project's license here, e.g., MIT License.]

## Acknowledgments

- Thanks to the researchers and developers whose work in bright-field imaging and segmentation (e.g., Residual U-Net approaches for RPE cells) has inspired this project.
- Special thanks to [aliaslandemir](https://github.com/aliaslandemir) for hosting the repository.

```
