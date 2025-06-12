import numpy as np
import cv2
import os
import json
import re

## CHANGED: Import PyTorch libraries
import torch
import torch.nn as nn
from torchvision import models
import glob
import torchvision.transforms as T
from sklearn.metrics import mean_absolute_error, accuracy_score


# * Configuration *
# Define in case the relative path to the images is not provided in the JSON
IMAGE_BASE_PATH = "."

# Path to the input JSON files
INPUT_FILE_PATH = './input.json'
OUTPUT_FILE_PATH = './output.json'

# Other important paths
CORNER_HORSE_TEMPLATE_PATH = "./cornerHorse_templates"
CLASSIFICATION_MODEL_PATH = './checkpoints_resnet50_1_squares/res_net_50_best_piece_classifier.pt'
REGRESSION_MODEL_PATH = './best_checkpoint_combination.pt'

# Definition of the device to use for PyTorch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# * Load Corner Horse Templates *
CORNER_HORSES_TEMPLATES = []
for filename in os.listdir(CORNER_HORSE_TEMPLATE_PATH):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        img_path = os.path.join(CORNER_HORSE_TEMPLATE_PATH, filename)
        img = cv2.imread(img_path)
        CORNER_HORSES_TEMPLATES.append(img)

# * Loading of the classification model *
def load_classification_model(path):
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)
    
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
        
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    
    return model

# * Define the ResNet50 model for multi-task learning *
class ResNet50MultiTask(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet50MultiTask, self).__init__()
        resnet = models.resnet50(pretrained=pretrained)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.flatten = nn.Flatten()
        self.presence_head = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        self.count_head = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Hardtanh(min_val=0, max_val=32)
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.flatten(features)
        presence_out = self.presence_head(features)
        count_out = self.count_head(features)
        return presence_out, count_out


# * Preprocessing for the CNN *
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# * Image Preprocessing functions *
def show_original_and_gray(image_path):
    original_img = cv2.imread(image_path)
    rgb_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    hsv_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
    lower_brown = np.array([2, 11, 11])
    upper_brown = np.array([60, 255, 255]) 
    wood_mask = cv2.inRange(hsv_img, lower_brown, upper_brown)
    wood_mask_inv = cv2.bitwise_not(wood_mask)
    no_wood_img = cv2.bitwise_and(original_img, original_img, mask=wood_mask_inv)
    
    gray_img = cv2.cvtColor(no_wood_img, cv2.COLOR_BGR2GRAY)
    return original_img, gray_img

def preprocess_image(gray_img, blur_kernel_size=17, intensity_factor=1.3, laplacian_kernel_size=3):
    blurred = cv2.GaussianBlur(gray_img, (blur_kernel_size, blur_kernel_size), 0)
    adjusted_img = cv2.convertScaleAbs(blurred, alpha=intensity_factor, beta=0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opened = cv2.morphologyEx(adjusted_img, cv2.MORPH_OPEN, kernel)
    laplacian = cv2.Laplacian(opened, cv2.CV_64F, ksize=laplacian_kernel_size)
    laplacian = cv2.convertScaleAbs(laplacian)
    otsu_thresh_val, _ = cv2.threshold(laplacian, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, binary_img = cv2.threshold(laplacian, otsu_thresh_val + 20, 255, cv2.THRESH_BINARY)
    return binary_img

def detect_lines(binary_img, min_line_length=50, max_line_gap=50):
    canny_image = cv2.Canny(binary_img, 50, 200)
    kernel = np.ones((13, 13), np.uint8)
    dilation_image = cv2.dilate(canny_image, kernel, iterations=1)
    lines = cv2.HoughLinesP(dilation_image, 1, np.pi / 180, threshold=500, 
                            minLineLength=min_line_length, maxLineGap=max_line_gap)
    black_image = np.zeros_like(dilation_image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(black_image, (x1, y1), (x2, y2), (255, 255, 255), 2)
    return black_image

def remove_noise_components(line_img, min_area=1000, keep_largest=True):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(line_img, connectivity=8)
    final_image = np.zeros_like(line_img)
    if keep_largest and num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_label = 1 + np.argmax(areas)
        final_image[labels == max_label] = 255
    else:
        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] >= min_area:
                final_image[labels == label] = 255
    return final_image

def find_chessboard_contour(line_img):
    contours, _ = cv2.findContours(line_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        return max(contours, key=cv2.contourArea)
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
        dst_corners = np.array([[0, 0], [board_size - 1, 0], [0, board_size - 1], [board_size - 1, board_size - 1]], dtype="float32")
        matrix = cv2.getPerspectiveTransform(ordered_corners, dst_corners)
        warped_original = cv2.warpPerspective(original_img, matrix, (board_size, board_size))
        warped_original_rgb = cv2.cvtColor(warped_original, cv2.COLOR_BGR2RGB)
        return warped_original_rgb, matrix
    else:
        print("Error: Did not find exactly 4 corners!")
        return None, None

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
        return None, None

def find_horse_template_matching(image_rgb, image_name=None):
    img_rgb = image_rgb.copy()
    height, width = img_rgb.shape[:2]
    best_match_val = -np.inf
    best_match_loc = None
    best_template_shape = None

    for template_rgb in CORNER_HORSES_TEMPLATES:
        if template_rgb is None: continue
        h, w = template_rgb.shape[:2]
        res = cv2.matchTemplate(img_rgb, template_rgb, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > best_match_val:
            best_match_val = max_val
            best_match_loc = max_loc
            best_template_shape = (h, w)
    
    if best_match_val >= 0.55:
        top_left = best_match_loc
        h, w = best_template_shape
        center_x, center_y = top_left[0] + w // 2, top_left[1] + h // 2
        corners = [("top-left", (0, 0)), ("top-right", (width, 0)), ("bottom-left", (0, height)), ("bottom-right", (width, height))]
        closest_corner = min(corners, key=lambda c: np.sqrt((center_x - c[1][0])**2 + (center_y - c[1][1])**2))
        if image_name: print(f"{image_name}: Found corner horse at {closest_corner[0]}")
        return closest_corner[0]
    else:
        if image_name: print(f"{image_name}: No good match found for corner horse.")
        return None

def detect_grid_lines(image_rgb, edge_threshold=20):
    height, width = image_rgb.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (17, 13), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=100, maxLineGap=10)
    if lines is None: return [], []

    horizontal_lines, vertical_lines = [], []
    for x1, y1, x2, y2 in lines[:, 0]:
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < 10: horizontal_lines.append((x1, y1, x2, y2))
        elif abs(angle - 90) < 10 or abs(angle + 90) < 10: vertical_lines.append((x1, y1, x2, y2))
    
    def merge_line_coords(lines, axis='y', threshold=30):
        if not lines: return []
        coords = sorted([int((l[1] + l[3]) / 2) if axis == 'y' else int((l[0] + l[2]) / 2) for l in lines])
        merged = []
        current = coords[0]
        for val in coords[1:]:
            if abs(val - current) < threshold: current = int((current + val) / 2)
            else: merged.append(current); current = val
        merged.append(current)
        return merged
    
    merged_horizontal = [y for y in merge_line_coords(horizontal_lines, 'y') if edge_threshold < y < (height - edge_threshold)]
    merged_vertical = [x for x in merge_line_coords(vertical_lines, 'x') if edge_threshold < x < (width - edge_threshold)]

    if len(merged_horizontal) != 7 or len(merged_vertical) != 7:
        print("Falling back to equally spaced grid lines")
        merged_horizontal = list(np.linspace(0, height, 9, dtype=int))[1:-1]
        merged_vertical = list(np.linspace(0, width, 9, dtype=int))[1:-1]
        
    return sorted(merged_horizontal), sorted(merged_vertical)

def run_regression(image_path, regression_model=None):
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Ensure RGB conversion
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        _, count_out = regression_model(img_tensor)
        count_rounded = torch.round(count_out)
    return int(count_rounded.item())

# * Main Execution *
if __name__ == "__main__":
    # Load classification model
    # print("Loading classification model...")
    # classification_model = load_classification_model(CLASSIFICATION_MODEL_PATH)
    
    print("Loading regression model...")
    regression_model = ResNet50MultiTask().to(DEVICE)
    checkpoint = torch.load(REGRESSION_MODEL_PATH, map_location=DEVICE, weights_only=True)
    regression_model.load_state_dict(checkpoint["model_state_dict"])
    regression_model.eval()

    # Read the input JSON file
    output_data = []
    count_failed = 0

    # Read image list
    try:
        with open(INPUT_FILE_PATH, 'r') as f:
            relative_image_files = json.load(f).get("image_files", [])
        print(f"Found {len(relative_image_files)} image files in {INPUT_FILE_PATH}")
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE_PATH} not found!")
        relative_image_files = []

    for relative_path in relative_image_files:
        image_path = os.path.join(IMAGE_BASE_PATH, relative_path)
        if not os.path.exists(image_path):
            print(f"Skipping non-existent file: {image_path}")
            continue

        filename = os.path.basename(image_path)
        print(f"--- Processing {filename} ---")

        # warped_original_rgb, matrix = process_chessboard_image(image_path)

        # if warped_original_rgb is None:
        #     count_failed += 1
        #     print(f"Failed {count_failed} times. Skipping...")
        #     print(f"Failed to detect chessboard in {filename}.")
        #     regressed_pieces = run_regression(image_path, regression_model)
        #     output_data.append({
        #         "image": filename,
        #         "error": "Could not detect chessboard.",
        #         "num_pieces": regressed_pieces
        #     })
        #     print(f"Regressed number of pieces: {regressed_pieces}")
        #     continue

        # closest_corner = find_horse_template_matching(warped_original_rgb, filename)
        # rotations = {"top-left": 90, "top-right": 180, "bottom-left": 0, "bottom-right": -90}
        # rotation_angle = rotations.get(closest_corner, 0)

        # (h, w) = warped_original_rgb.shape[:2]
        # center = (w // 2, h // 2)
        # M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        # warped_rotated_rgb = cv2.warpAffine(warped_original_rgb, M, (w, h))

        # board_size = warped_rotated_rgb.shape[0]
        # crop_margin = int(board_size * 0.07)
        # cropped_board = warped_rotated_rgb[crop_margin:board_size-crop_margin, crop_margin:board_size-crop_margin]

        # h_lines, v_lines = detect_grid_lines(cropped_board)

        # if len(h_lines) != 7 or len(v_lines) != 7:
        #     print(f"Failed to detect a valid 8x8 grid in {filename}.")
        #     regressed_pieces = run_regression(image_path, regression_model)
        #     output_data.append({
        #         "image": filename,
        #         "error": "Could not detect 8x8 grid.",
        #         "num_pieces": regressed_pieces
        #     })
        #     print(f"Regressed number of pieces: {regressed_pieces}")
        #     continue

        # # Classification temporarily disabled
        # height, width = cropped_board.shape[:2]
        # h_lines_full = [0] + h_lines + [height]
        # v_lines_full = [0] + v_lines + [width]

        # squares = []
        # for i in range(8):
        #     for j in range(8):
        #         y1, y2 = h_lines_full[i], h_lines_full[i + 1]
        #         x1, x2 = v_lines_full[j], v_lines_full[j + 1]
        #         square_img = cropped_board[y1:y2, x1:x2]
        #         squares.append(square_img)

        # batch_tensors = [transform(img) for img in squares]
        # input_batch = torch.stack(batch_tensors).to(DEVICE)

        # with torch.no_grad():
        #     logits = classification_model(input_batch)
        #     probs = torch.sigmoid(logits).squeeze(1)
        #     predictions = (probs > 0.5).int()

        # num_pieces = predictions.sum().item()
        # print(f"Detected {num_pieces} pieces in {filename}")

        # if num_pieces > 32:
        #     regressed_pieces = run_regression(image_path, regression_model)
        #     num_pieces = regressed_pieces
        #     print(f"Warning: Detected >32 pieces. Using regressed value: {regressed_pieces}")

        # Only use regression for now
        regressed_pieces = run_regression(image_path, regression_model)
        print(f"Regressed number of pieces: {regressed_pieces}")
        result = {"image": filename, "num_pieces": regressed_pieces}

        output_data.append(result)

    # Save results
    with open(OUTPUT_FILE_PATH, 'w') as f:
        json.dump(output_data, f, indent=4)
        
    # Calculate MAE and accuracy using ground truth from training_inputs/matrices
    gt_dir = '../Shared/training_inputs/matrices'
    y_true = []
    y_pred = []

    for result in output_data:
        img_file = result["image"]
        img_name = os.path.splitext(img_file)[0]  # e.g., G000_IMG000
        pred = result.get("num_pieces")

        gt_path = os.path.join(gt_dir, f"{img_name}.json")
        if not os.path.isfile(gt_path):
            print(f"Warning: Ground truth not found for {img_name}")
            continue

        with open(gt_path, 'r') as f:
            gt_data = json.load(f)

        # Get the true number of pieces
        gt = None
        if "piece_count" in gt_data:
            gt = gt_data["piece_count"]
            
        if gt is not None and pred is not None:
            y_true.append(gt)
            y_pred.append(pred)
            # print(f"{img_name}: Predicted = {pred}, Ground Truth = {gt}")
        else:
            print(f"Invalid prediction or ground truth for {img_name}")

    # Final metrics
    if y_true and y_pred:
        print("True piece counts for test set: ", y_true)
        print("Predicted piece counts for test set: ", y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        print(f"\nMAE: {mae:.3f}")
        print(f"Accuracy: {accuracy:.3%}")
    else:
        print("No valid ground truth comparisons could be made.")

    print(f"\nProcessing complete. Output written to {OUTPUT_FILE_PATH}")