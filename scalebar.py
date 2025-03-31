import cv2
import numpy as np
import math

# ------------------------------------------------------------------
# Helper: Draw text with a background rectangle (for readability)
# ------------------------------------------------------------------
def draw_text_with_bg(
    image, 
    text, 
    origin, 
    font=cv2.FONT_HERSHEY_SIMPLEX, 
    font_scale=1.0, 
    text_color=(255,255,255), 
    bg_color=(0,0,0), 
    thickness=2, 
    padding=6
):
    """
    Draw text with a solid background for better visibility.
    """
    x, y = origin
    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size
    
    # Background rectangle corners
    bg_tl = (x, y - text_h - padding // 2)
    bg_br = (x + text_w + padding, y + baseline + padding // 2)
    
    # Draw background
    cv2.rectangle(image, bg_tl, bg_br, bg_color, -1)
    # Draw text on top
    cv2.putText(image, text, (x + padding // 2, y), font, font_scale, text_color, thickness)

# ------------------------------------------------------------------
# Helper: Annotate the original image with scale info and scale bar
# ------------------------------------------------------------------
def add_scale_info(image, um_per_px, px_per_um, scale_bar_length_px, scale_bar_label):
    """
    Overlays scale info (um/px, px/um) and a large scale bar on the image.
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]
    
    # 1) um/px (top-left)
    text1 = f"um/px: {um_per_px:.3f}"
    draw_text_with_bg(annotated, text1, (10, 40), font_scale=1.0)
    
    # 2) px/um (top-right)
    text2 = f"px/um: {px_per_um:.3f}"
    text_size, baseline = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    text_w, text_h = text_size
    x_pos = w - text_w - 10
    y_pos = 40
    draw_text_with_bg(annotated, text2, (x_pos, y_pos), font_scale=1.0)
    
    # 3) Scale bar in bottom-left
    bar_margin = 50
    bar_y = h - bar_margin
    bar_x_start = bar_margin
    bar_x_end = bar_x_start + scale_bar_length_px
    
    # Ensure bar doesn't exceed image width
    if bar_x_end > (w - bar_margin):
        bar_x_end = w - bar_margin
    
    # Draw the bar (thick white line)
    cv2.line(annotated, (bar_x_start, bar_y), (bar_x_end, bar_y), (255, 255, 255), 6)
    
    # Label above the bar
    label_x = bar_x_start
    label_y = bar_y - 15
    draw_text_with_bg(annotated, scale_bar_label, (label_x, label_y), font_scale=1.0)
    
    return annotated

# ------------------------------------------------------------------
# Main function: ROI selection, zoomable view, distance measurement
# ------------------------------------------------------------------
def main():
    # -----------------------------
    # 1) Load image
    # -----------------------------
    image_path = "20x.jpg"  # Replace with your file
    img = cv2.imread(image_path)
    if img is None:
        print("Error loading image!")
        return
    h_full, w_full = img.shape[:2]
    
    # -----------------------------
    # 2) User selects ROI on full image
    # -----------------------------
    print("Draw a box around the region of interest, then press ENTER or SPACE.")
    roi = cv2.selectROI("Select ROI", img, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select ROI")
    
    x_roi, y_roi, w_roi, h_roi = roi
    if w_roi == 0 or h_roi == 0:
        print("No ROI selected. Exiting.")
        return
    
    # Crop the ROI
    roi_img = img[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
    roi_h, roi_w = roi_img.shape[:2]
    
    # -----------------------------
    # 3) Zoomable window for the ROI
    #    - You can pan with arrow keys / WASD
    #    - Zoom with trackbar
    #    - Click 2 points to measure
    # -----------------------------
    zoom_factor = 1
    max_zoom = 10
    offset_x, offset_y = 0, 0
    points = []
    done_selecting = False
    
    # Mouse callback: record clicked points (in ROI coordinates)
    def mouse_callback(event, x, y, flags, param):
        nonlocal points, done_selecting
        if event == cv2.EVENT_LBUTTONDOWN:
            # Map display coords back to ROI coords
            roi_x = int(offset_x + x / zoom_factor)
            roi_y = int(offset_y + y / zoom_factor)
            # Ensure it's within the ROI bounds
            roi_x = max(0, min(roi_x, roi_w-1))
            roi_y = max(0, min(roi_y, roi_h-1))
            
            points.append((roi_x, roi_y))
            if len(points) == 2:
                done_selecting = True
    
    # Trackbar callback
    def on_trackbar(val):
        nonlocal zoom_factor
        zoom_factor = max(1, val)
    
    # Create a window & trackbar
    cv2.namedWindow("Zoom ROI", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Zoom ROI", roi_w, roi_h)
    cv2.createTrackbar("Zoom", "Zoom ROI", zoom_factor, max_zoom, on_trackbar)
    cv2.setMouseCallback("Zoom ROI", mouse_callback)
    
    # Function to create the displayed (zoomed/panned) ROI
    def get_display_image():
        nonlocal offset_x, offset_y, zoom_factor
        # Size of the visible region in the ROI (unscaled)
        view_w = int(roi_w / zoom_factor)
        view_h = int(roi_h / zoom_factor)
        
        # Clamp offset
        offset_x = max(0, min(offset_x, roi_w - view_w))
        offset_y = max(0, min(offset_y, roi_h - view_h))
        
        # Extract that sub-ROI
        sub_roi = roi_img[offset_y:offset_y+view_h, offset_x:offset_x+view_w]
        
        # Scale it up to fill the window
        disp = cv2.resize(sub_roi, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)
        
        # Draw any selected points or lines
        tmp = disp.copy()
        if len(points) > 0:
            for p in points:
                # Map ROI coords to display coords
                disp_x = int((p[0] - offset_x) * zoom_factor)
                disp_y = int((p[1] - offset_y) * zoom_factor)
                cv2.circle(tmp, (disp_x, disp_y), 5, (0, 255, 0), -1)
        if len(points) == 2:
            p1 = (int((points[0][0] - offset_x)*zoom_factor), int((points[0][1] - offset_y)*zoom_factor))
            p2 = (int((points[1][0] - offset_x)*zoom_factor), int((points[1][1] - offset_y)*zoom_factor))
            cv2.line(tmp, p1, p2, (255, 0, 0), 2)
        
        return tmp
    
    print("Instructions for ROI Zoom Window:")
    print("  - Use the trackbar (top) to zoom in/out.")
    print("  - Use arrow keys or WASD to pan.")
    print("  - Click two points to measure.")
    print("  - Press ESC to cancel, or close window when done.")
    
    while True:
        disp_img = get_display_image()
        cv2.imshow("Zoom ROI", disp_img)
        key = cv2.waitKey(20) & 0xFF
        
        if done_selecting:  # Two points clicked
            break
        if key == 27:  # ESC
            cv2.destroyWindow("Zoom ROI")
            print("Measurement canceled.")
            return
        # Pan controls
        step = max(1, int(20 / zoom_factor))
        if key in [ord('a'), 81]:  # left arrow or 'a'
            offset_x -= step
        elif key in [ord('d'), 83]:  # right arrow or 'd'
            offset_x += step
        elif key in [ord('w'), 82]:  # up arrow or 'w'
            offset_y -= step
        elif key in [ord('s'), 84]:  # down arrow or 's'
            offset_y += step
    
    cv2.destroyWindow("Zoom ROI")
    if len(points) < 2:
        print("Not enough points selected. Exiting.")
        return
    
    # -----------------------------
    # 4) Compute distance in original image coords
    # -----------------------------
    # ROI coords => global coords
    p1_roi = points[0]
    p2_roi = points[1]
    p1_global = (p1_roi[0] + x_roi, p1_roi[1] + y_roi)
    p2_global = (p2_roi[0] + x_roi, p2_roi[1] + y_roi)
    
    pixel_distance = math.hypot(p2_global[0] - p1_global[0], p2_global[1] - p1_global[1])
    print(f"Measured line length: {pixel_distance:.2f} pixels")
    
    # -----------------------------
    # 5) Ask for known distance (in um) -> calibrate
    # -----------------------------
    known_distance_str = input("Enter the known distance (in um) for the measured line pair: ")
    try:
        known_distance = float(known_distance_str)
    except ValueError:
        print("Invalid number. Exiting.")
        return
    
    um_per_px = known_distance / pixel_distance
    px_per_um = pixel_distance / known_distance
    print(f"Calculated scale: {um_per_px:.4f} um/px  ({px_per_um:.4f} px/um)")
    
    # -----------------------------
    # 6) Annotate the original image with scale info
    # -----------------------------
    scale_bar_length_um = known_distance
    scale_bar_length_px = int(scale_bar_length_um / um_per_px)
    scale_bar_label = f"{scale_bar_length_um:.1f} um"
    
    annotated = add_scale_info(img, um_per_px, px_per_um, scale_bar_length_px, scale_bar_label)
    
    # Optionally draw the measured line in red
    cv2.line(annotated, p1_global, p2_global, (0, 0, 255), 2)
    
    # Show and save
    cv2.imshow("Annotated Image", annotated)
    print("Press any key to close the annotated image window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("annotated_image.jpg", annotated)
    print("Annotated image saved as annotated_image.jpg.")

if __name__ == "__main__":
    main()
