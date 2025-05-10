import numpy as np
import cv2
import os
import xgboost as xgb
import json
import re

# Path to the input JSON file containing image filenames to process
INPUT_FILE_PATH = './input.json'
# Path to the output JSON file where results will be saved
OUTPUT_FILE_PATH = './output.json'
# Path to the directory containing corner horse templates
CORNER_HORSE_TEMPLATE_PATH = "./cornerHorse_templates/"

# Load the images for the corner horses and store them in a list
CORNER_HORSES_TEMPLATES = []

for filename in os.listdir(CORNER_HORSE_TEMPLATE_PATH):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        img = cv2.imread(os.path.join(CORNER_HORSE_TEMPLATE_PATH, filename))
        CORNER_HORSES_TEMPLATES.append(img)
    else:
        continue

def show_original_and_gray(image_path):
    original_img = cv2.imread(image_path)
    rgb_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # Wood removal by color thresholding (targeting brown/wooden colors)
    hsv_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
    
    # Define range for brown/wooden colors in HSV
    lower_brown = np.array([2, 11, 11])
    upper_brown = np.array([60, 255, 255]) 
    
    # Create mask for wood
    wood_mask = cv2.inRange(hsv_img, lower_brown, upper_brown)
    
    # Invert the mask to keep non-wood parts
    wood_mask_inv = cv2.bitwise_not(wood_mask)
    
    # Apply the mask to the original image
    no_wood_img = cv2.bitwise_and(original_img, original_img, mask=wood_mask_inv)
    
    
    # Convert to grayscale
    gray_img = cv2.cvtColor(no_wood_img, cv2.COLOR_BGR2GRAY)
    
    return original_img, gray_img

def preprocess_image(gray_img, blur_kernel_size=17, intensity_factor=1.3, laplacian_kernel_size=3):
    blurred = cv2.GaussianBlur(gray_img, (blur_kernel_size, blur_kernel_size), 0)
    adjusted_img = cv2.convertScaleAbs(blurred, alpha=intensity_factor, beta=0)

    # Morphological opening to remove small details like pieces
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opened = cv2.morphologyEx(adjusted_img, cv2.MORPH_OPEN, kernel)

    laplacian = cv2.Laplacian(opened, cv2.CV_64F, ksize=laplacian_kernel_size)
    laplacian = cv2.convertScaleAbs(laplacian)

    # OTSU + optional manual offset to suppress weak edges
    otsu_thresh_val, _ = cv2.threshold(laplacian, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, binary_img = cv2.threshold(laplacian, otsu_thresh_val + 20, 255, cv2.THRESH_BINARY)

    return binary_img

def detect_lines(binary_img, min_line_length=50, max_line_gap=50):
    # Canny edge detection with lower threshold
    canny_image = cv2.Canny(binary_img, 50, 200)  # Tuning thresholds to capture better edges

    # Use dilation to reinforce edges
    kernel = np.ones((13, 13), np.uint8)
    dilation_image = cv2.dilate(canny_image, kernel, iterations=1)
    
    # Hough Lines transform for line detection
    lines = cv2.HoughLinesP(dilation_image, 1, np.pi / 180, threshold=500, 
                            minLineLength=min_line_length, maxLineGap=max_line_gap)

    # Create an image to store the detected lines
    black_image = np.zeros_like(dilation_image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(black_image, (x1, y1), (x2, y2), (255, 255, 255), 2)
    
    return black_image

def remove_noise_components(line_img, min_area=1000, keep_largest=True):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(line_img, connectivity=8)
    
    # Create a black image to store the final result
    final_image = np.zeros_like(line_img)
    
    if keep_largest:
        # Get the areas of all components
        areas = stats[1:, cv2.CC_STAT_AREA]
        
        # Find the label of the largest component
        max_label = 1 + np.argmax(areas)
        
        # Keep only the largest component
        final_image[labels == max_label] = 255
    else:
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= min_area:
                final_image[labels == label] = 255

    return final_image

def find_chessboard_contour(line_img):
    contours, _ = cv2.findContours(line_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    if contours:
        return contours[0]
    return None

def order_corners(corners):
    sorted_by_y = sorted(corners, key=lambda p: p[1])
    top_two = sorted(sorted_by_y[:2], key=lambda p: p[0])
    bottom_two = sorted(sorted_by_y[2:], key=lambda p: p[0])
    return np.array([top_two[0], top_two[1], bottom_two[0], bottom_two[1]], dtype="float32")

def warp_chessboard(gray_img, original_img, contour, board_size=800):
    epsilon = 0.05 * cv2.arcLength(contour, True)
    approx_corners = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx_corners) == 4:
        corners = np.squeeze(approx_corners)
        ordered_corners = order_corners(corners)
        dst_corners = np.array([
            [0, 0], [board_size - 1, 0],
            [0, board_size - 1], [board_size - 1, board_size - 1]
        ], dtype="float32")
        matrix = cv2.getPerspectiveTransform(ordered_corners, dst_corners)
        warped_board = cv2.warpPerspective(gray_img, matrix, (board_size, board_size))
        
        # Also warp the original RGB image
        warped_original = cv2.warpPerspective(original_img, matrix, (board_size, board_size))
        warped_original_rgb = cv2.cvtColor(warped_original, cv2.COLOR_BGR2RGB)
        
        # Return the warp matrix as well to allow for reverting the transform
        return warped_board, warped_original_rgb, matrix, ordered_corners, dst_corners
    else:
        print("Error: Did not find exactly 4 corners!")
        return None, None, None, None, None
    
def process_chessboard_image(image_path):
    original_img, gray_img = show_original_and_gray(image_path)
    otsu_binary = preprocess_image(gray_img)
    line_img = detect_lines(otsu_binary)
    clean_line_img = remove_noise_components(line_img)
    contour = find_chessboard_contour(clean_line_img)
    if contour is not None:
        return warp_chessboard(gray_img, original_img, contour)
    else:
        print("Chessboard contour not found!")
        return None
    
def detect_grid_lines(image_rgb, show_lines=False, edge_threshold=20):
    height, width = image_rgb.shape[:2]

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (17, 13), 0)

    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=100, maxLineGap=10)
    
    if lines is None:
        print("No lines found")
        return [], []

    horizontal_lines = []
    vertical_lines = []

    for x1, y1, x2, y2 in lines[:, 0]:
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < 10:
            horizontal_lines.append((x1, y1, x2, y2))
        elif abs(angle - 90) < 10 or abs(angle + 90) < 10:
            vertical_lines.append((x1, y1, x2, y2))

    def merge_line_coords(lines, axis='y', threshold=30):
        if not lines:
            return []
        coords = [int((y1 + y2) / 2) if axis == 'y' else int((x1 + x2) / 2) for x1, y1, x2, y2 in lines]
        coords = sorted(coords)
        merged = []
        current = coords[0]
        for val in coords[1:]:
            if abs(val - current) < threshold:
                current = int((current + val) / 2)
            else:
                merged.append(current)
                current = val
        merged.append(current)
        return merged

    merged_horizontal = merge_line_coords(horizontal_lines, axis='y')
    merged_vertical = merge_line_coords(vertical_lines, axis='x')

    # Remove lines too close to image edges
    merged_horizontal = [y for y in merged_horizontal if edge_threshold < y < (height - edge_threshold)]
    merged_vertical = [x for x in merged_vertical if edge_threshold < x < (width - edge_threshold)]

    filled_horizontal = []
    filled_vertical = []

    def fill_missing_lines(existing, filled, total_lines, size, tolerance=40):
        print(f"Filling missing lines: {len(existing)} found, {total_lines} expected")
        ideal_positions = list(np.linspace(0, size, total_lines + 2, dtype=int)[1:-1])
        for pos in ideal_positions:
            if not any(abs(pos - e) < tolerance for e in existing):
                existing.append(pos)
                filled.append(pos)
        existing.sort()
        return existing

    if len(merged_horizontal) < 7:
        merged_horizontal = fill_missing_lines(merged_horizontal, filled_horizontal, 7, height)
    if len(merged_vertical) < 7:
        merged_vertical = fill_missing_lines(merged_vertical, filled_vertical, 7, width)
        
    # If we don't have exactly 7 lines in each direction, use equally spaced lines as fallback
    if len(merged_horizontal) != 7 or len(merged_vertical) != 7:
        print(f"Insufficient grid lines detected: {len(merged_horizontal)} horizontal, {len(merged_vertical)} vertical")
        print("Falling back to equally spaced grid lines")

        # Generate 9 lines (including borders)
        horizontal_all = list(np.linspace(0, height, 9, dtype=int))
        vertical_all = list(np.linspace(0, width, 9, dtype=int))

        # Take internal 7 lines (excluding the first and last, which are borders)
        merged_horizontal = horizontal_all[1:-1]
        merged_vertical = vertical_all[1:-1]

        # Mark all as filled (fallback)
        filled_horizontal = merged_horizontal.copy()
        filled_vertical = merged_vertical.copy()

    return merged_horizontal, merged_vertical

def find_horse_template_matching(image_rgb, image_name=None):
    img_rgb = image_rgb.copy()
    height, width = img_rgb.shape[:2]
    
    best_match_val = -np.inf
    best_match_loc = None
    best_template_shape = None
    best_template_index = None

    for idx, template_rgb in enumerate(CORNER_HORSES_TEMPLATES):
        if template_rgb is None:
            print(f"Template at index {idx} is None.")
            continue

        h, w = template_rgb.shape[:2]
        
        # Template matching
        res = cv2.matchTemplate(img_rgb, template_rgb, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        if max_val > best_match_val:
            best_match_val = max_val
            best_match_loc = max_loc
            best_template_shape = (h, w)
            best_template_index = idx

    print(f"Best match index: {best_template_index} with score {best_match_val:.2f}")
    if best_match_val >= 0.55:
        top_left = best_match_loc
        h, w = best_template_shape
        bottom_right = (top_left[0] + w, top_left[1] + h)

        # Draw match rectangle
        cv2.rectangle(img_rgb, top_left, bottom_right, (0, 255, 0), 2)

        # Center of match
        center_x = top_left[0] + w // 2
        center_y = top_left[1] + h // 2

        # Determine closest corner
        corners = [
            ("top-left", (0, 0)),
            ("top-right", (width, 0)),
            ("bottom-left", (0, height)),
            ("bottom-right", (width, height))
        ]
        
        closest_corner = min(corners, key=lambda c: 
            np.sqrt((center_x - c[1][0])**2 + (center_y - c[1][1])**2))

        # Print image name and closest corner
        if image_name:
            print(f"{image_name}: {closest_corner[0]}")
        else:
            print(f"Closest corner: {closest_corner[0]}")
            
        return closest_corner[0]
    else:
        if image_name:
            print(f"{image_name}: No good match found.")
        else:
            print("No good match found.")
        return None
    
def detect_circles(grey_board, min_radius=20, max_radius=32):
    grey_board = cv2.GaussianBlur(grey_board, (5, 5), 0)
    circles = cv2.HoughCircles(grey_board, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                                param1=65, param2=20, minRadius=min_radius, maxRadius=max_radius)
    if circles is not None:
        return np.round(circles[0, :]).astype("int")
    return []

def extract_features(image):
    h, w, _ = image.shape

    # Grayscale version for variance and edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Center region
    center = image[h//4:3*h//4, w//4:3*w//4]
    avg_center_color = np.mean(center)

    # Full square stats
    avg_square_color = np.mean(image)
    color_contrast_center = abs(avg_center_color - avg_square_color)
    color_variance = np.var(image)

    # Top vs bottom
    top_half = image[:h//2, :]
    bottom_half = image[h//2:, :]
    mean_top = np.mean(top_half)
    mean_bottom = np.mean(bottom_half)
    contrast_top_bottom = abs(mean_top - mean_bottom)

    # Edge strength using Sobel
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
    edge_strength = np.mean(np.abs(sobel))

    return [
        avg_center_color,
        avg_square_color,
        color_contrast_center,
        color_variance,
        mean_top,
        mean_bottom,
        contrast_top_bottom,
        edge_strength
    ]

def validate_with_model(image):
    # Extract features from the image
    features = np.array(extract_features(image), dtype=np.float32).reshape(1, -1)  # Ensure it's a 2D array (1, num_features)
    
    # Predict using the model
    prediction_prob = xgb_model.predict(features)[0]
    
    # 1 = piece, 0 = empty
    return prediction_prob

def map_points_back_to_original(points, full_transform_matrix):
    # Compute the inverse of the full transformation matrix
    inverse_matrix = np.linalg.inv(full_transform_matrix)

    # Convert points to the required shape (N, 1, 2)
    points_array = np.array(points, dtype='float32').reshape(-1, 1, 2)

    # Apply the inverse perspective transform
    mapped = cv2.perspectiveTransform(points_array, inverse_matrix)

    # Reshape the result to (N, 2)
    return mapped.reshape(-1, 2).astype(int)

def process_image(key, v):
    board_size = v['warped_original_rgb'].shape[0]
    crop_margin = int(board_size * 0.07)

    cropped_board = v['warped_original_rgb'][crop_margin:board_size - crop_margin,
                                             crop_margin:board_size - crop_margin]
    v['cropped_board'] = cropped_board

    h_lines, v_lines = detect_grid_lines(cropped_board, show_lines=False)

    if len(h_lines) != 7 or len(v_lines) != 7:
        print(f"Not enough grid lines detected in {key}.")
        return None

    height, width = cropped_board.shape[:2]
    h_lines_full = [0] + sorted(h_lines) + [height]
    v_lines_full = [0] + sorted(v_lines) + [width]

    squares = []
    square_centers = []
    square_bounds = []

    for i in range(8):
        for j in range(8):
            y1, y2 = h_lines_full[i], h_lines_full[i + 1]
            x1, x2 = v_lines_full[j], v_lines_full[j + 1]
            square = cropped_board[y1:y2, x1:x2]
            rank = 8 - i
            file = files[j]
            label = f"{file}{rank}"
            squares.append({"position": (i, j), "label": label, "image": square})
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            square_centers.append((center_x, center_y))
            square_bounds.append((x1, y1, x2, y2))

    v['squares'] = squares

    gray = cv2.cvtColor(cropped_board, cv2.COLOR_RGB2GRAY)
    circles = detect_circles(gray)

    validated_circles = []
    rejected_circles = []

    board_matrix = []
    for i in range(len(square_centers)):
        x1, y1, x2, y2 = square_bounds[i]
        presence = 0
        for (x, y, r) in circles:
            if (x1 <= x <= x2 and y1 <= y <= y2 and 
                x - r + BORDER_TOLERANCE >= x1 and x + r - BORDER_TOLERANCE <= x2 and 
                y - r + BORDER_TOLERANCE >= y1 and y + r - BORDER_TOLERANCE <= y2):
                
                square_img = cropped_board[y1:y2, x1:x2]
                if validate_with_model(square_img):  # validated by model
                    validated_circles.append((x, y, r))
                    presence = 1
                    break
                else:
                    rejected_circles.append((x, y, r))

        board_matrix.append(presence)

    board_matrix_2d = np.array(board_matrix).reshape(8, 8)
    v['presence_matrix'] = board_matrix_2d

    print(f"Presence Matrix for {key}:")
    print("Number of pieces detected:", np.sum(board_matrix_2d))
    print(np.flipud(board_matrix_2d).tolist())

    # Map bounding boxes back to original image
    detected_boxes = []
    
    adjusted_circles = []
    for (x, y, r) in validated_circles:
        adjusted_x = x + crop_margin
        adjusted_y = y + crop_margin
        adjusted_circles.append((adjusted_x, adjusted_y, r))

    centers = np.float32([
        [[x, y]] for (x, y, r) in adjusted_circles
    ])  # shape: (N, 1, 2)
    
    circle_centers_original  = map_points_back_to_original(centers, v['matrix_full'])
    v['detected_pieces'] = circle_centers_original 

    # Save the coordinates of the boxes
    for (x, y) in circle_centers_original:
        detected_boxes.append((x - 100, y - 160, x + 90, y + 110))
    v['bounding_boxes'] = detected_boxes
    
    return v

# Read the input.json file to get the list of image files
try:
    with open(INPUT_FILE_PATH, 'r') as input_file:
        input_data = json.load(input_file)
        image_files = input_data.get("image_files", [])
        print(f"Found {len(image_files)} image files in input.json")
except FileNotFoundError:
    print(f"Error: {INPUT_FILE_PATH} not found!")
    image_files = []

# Ensure only existing files are processed
image_files = [f for f in image_files if os.path.exists(f)]
print(f"Processing {len(image_files)} valid image files from input.json")

# Process each image
failed_images = []
rotations = {
    "top-left": 90,
    "top-right": 180,
    "bottom-left": 0,
    "bottom-right": -90
}

successful_images = {}

for image_path in image_files:
    filename = os.path.basename(image_path)
    print(f"Processing {filename}")

    original_img, _ = show_original_and_gray(image_path)
    warped, warped_original_rgb, matrix, ordered_corners, dst_corners = process_chessboard_image(image_path)

    if warped is not None:
        # Detect orientation
        closest_corner = find_horse_template_matching(warped_original_rgb, filename)
        rotation_angle = rotations.get(closest_corner, 0)
        
        # Create rotation matrix in homogeneous coordinates (3x3)
        center = (800 // 2, 800 // 2)
        R_affine = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)  # 2x3
        R_homogeneous = np.vstack([R_affine, [0, 0, 1]])  # Convert to 3x3

        # Combine with warp matrix (H: 3x3)
        H_full = R_homogeneous @ matrix  # Final matrix from original image to rotated warped
        H_inv = np.linalg.inv(H_full)    # For going back
        # Apply rotation to both warped images
        def rotate_image(image, angle):
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        warped_rotated = rotate_image(warped, rotation_angle)
        warped_original_rgb_rotated = rotate_image(warped_original_rgb, rotation_angle)

        successful_images[filename] = {
            'original image': original_img,
            'warped': warped_rotated,
            'warped_original_rgb': warped_original_rgb_rotated,
            'matrix_full': H_full,
            'matrix_inverse': H_inv,
            'ordered_corners': ordered_corners,
            'dst_corners': dst_corners
        }

    else:
        failed_images.append(filename)

# Summary
print("\nImages where chessboard detection failed:")
for failed_image in failed_images:
    print(failed_image)

print(f"\nSuccessfully processed {len(image_files) - len(failed_images)}/{len(image_files)} images")

files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

for k, v in successful_images.items():
    board_size = v['warped_original_rgb'].shape[0]
    crop_margin = int(board_size * 0.07)

    cropped_board = v['warped_original_rgb'][crop_margin:board_size - crop_margin,
                                             crop_margin:board_size - crop_margin]
    v['cropped_board'] = cropped_board
    h_lines, v_lines = detect_grid_lines(cropped_board, show_lines=False)

    if len(h_lines) != 7 or len(v_lines) != 7:
        print(f"Not enough grid lines detected in {k}.")
        continue

    height, width = cropped_board.shape[:2]
    h_lines_full = [0] + sorted(h_lines) + [height]
    v_lines_full = [0] + sorted(v_lines) + [width]

    squares = []

    for i in range(8):  # rows (0 = top = rank 8)
        for j in range(8):  # columns (0 = left = file a)
            y1, y2 = h_lines_full[i], h_lines_full[i + 1]
            x1, x2 = v_lines_full[j], v_lines_full[j + 1]
            square = cropped_board[y1:y2, x1:x2]
            rank = 8 - i
            file = files[j]
            label = f"{file}{rank}"
            squares.append({
                "position": (i, j),
                "label": label,
                "image": square
            })

    v['squares'] = squares

DISTANCE_THRESHOLD = 25
BORDER_TOLERANCE = 5

# Load the XGBoost model
xgb_model = xgb.XGBClassifier()
xgb_model.load_model("xgb_piece_detector.json")

presence_matrix = []

for k, v in successful_images.items():
    result = process_image(k, v)
    if result:
        presence_matrix.append(result['presence_matrix'])

# Prepare the data for the JSON file
output_data = []

for k, v in successful_images.items():
    presence_matrix = v.get('presence_matrix', [])
    detected_pieces = v.get('bounding_boxes', [])

    # Convert bounding boxes to the required format
    detected_pieces_formatted = [
        {
            "xmin": int(box[0]),
            "ymin": int(box[1]),
            "xmax": int(box[2]),
            "ymax": int(box[3])
        }
        for box in detected_pieces
    ]

    output_data.append({
        "image": k,
        "num_pieces": int(np.sum(presence_matrix)),  # Convert to native Python int
        "board": np.flipud(presence_matrix).tolist(),  # Convert to native Python list
        "detected_pieces": detected_pieces_formatted
    })

# Dump the JSON string with indentation
json_string = json.dumps(output_data, indent=4)

# Post-process to compact "board" values only (pretty formatting)
def compact_board_arrays(match):
    board_content = match.group(1)
    # Find all rows and join them properly
    rows = re.findall(r'\[\s*([0-1,\s]+?)\s*\]', board_content)
    # Remove unnecessary spaces and join rows inline, ensuring proper format
    compact_rows = [f"    [{' '.join(row.strip().split())}]" for row in rows]
    return f'"board": [\n        ' + ',\n        '.join(compact_rows) + '\n\t    ]'

# Apply only to "board": [...]
json_string = re.sub(
    r'"board": \[\s*((?:\s*\[\s*[0-1,\s]+\s*\],?\s*)+)\s*\]',
    compact_board_arrays,
    json_string
)

# Write the final formatted string to file
with open(OUTPUT_FILE_PATH, 'w') as f:
    f.write(json_string)

print(f"Output written to {OUTPUT_FILE_PATH}")

for image in failed_images:
    print(f"Failed to process image: {image}")