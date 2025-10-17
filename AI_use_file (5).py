#This python code uses an AI and some post processing to mask ultrasound video.

#This code only works on processed mat files. Files can be processed through the script "process_image_mat_files.py"

#About the rectangle you draw to create the ROI (Region of interest):
# - The width of the rectangle effects nothing (the width is automatically set to the max).
#The height of the rectangle is the only part that matters when drawing the rectangle.
# - You can choose not to create a ROI by just closing napari at the start when it opens.

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from scipy.io import loadmat
import os
import napari
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import label
from scipy.ndimage import binary_fill_holes
from scipy.interpolate import UnivariateSpline
from scipy.io import savemat
from magicgui import magicgui
from qtpy.QtCore import QTimer


def get_files(folder):
    return [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

#The directory of the processed ultrasound footage || CHANGE THESE TO YOUR DIRECTORIES
image_dir = 'Segmentation\\Processed_img'
save_dir = 'Segmentation\\Mask_output'

files_a = get_files(image_dir)

#the name of the file to predict a mask for
file_name = "V04 - right Internal jugular vein - longitudinal - caudal - 0 g (1)_HRI_PW.mat"

#First model named "AI_segmentation_unet.h5" #11 training videos #1668 frames per
#second model named "AI_segmentation_unet_2_667_frames_per.h5" #31 training videos
#Third model named "AI_segmentation_unet_3_1267_frames_per.h5" #31 training videos
#fourth model named "AI_segmentation_unet_4_1267_frames_double_filters.h5" #31 training videos
#fifth model named "AI_segmentation_unet_5_1267_frames_double_filters.h5" #33 training videos
#sixth model named "AI_segmentation_unet_6_1267_frames_per.h5" #33 training videos

#After testing a bunch of different footage model 3 and 6 seem to be the best
#If the result is bad try switching the model or using different post processing

model_names = ["AI_segmentation_unet.h5","AI_segmentation_unet_2_667_frames_per.h5","AI_segmentation_unet_3_1267_frames_per.h5","AI_segmentation_unet_4_1267_frames_double_filters.h5","AI_segmentation_unet_5_1267_frames_double_filters.h5","AI_segmentation_unet_6_1267_frames_per.h5"]

#change this number to 0-5 to change which AI model is being used
model_name = model_names[5]

#Load the trained AI model
model = load_model(model_name)

def keep_largest_component(mask):
    """Keep only the largest connected component in the binary mask."""
    labeled_array, num_features = label(mask)
    if num_features == 0:
        return mask

    # Count pixels in each component
    component_sizes = np.bincount(labeled_array.ravel())
    component_sizes[0] = 0  # ignore background
    largest_label = component_sizes.argmax()

    return (labeled_array == largest_label).astype(np.uint8) 

def fill_mask_holes(mask):
    """Fill internal holes in the binary mask."""
    return binary_fill_holes(mask).astype(np.uint8)

def connect_vertical_edges(mask):
    """Ensure the leftmost and rightmost edges are connected vertically,
    and connect edge extremities to the nearest internal mask pixel."""
    
    h, w = mask.shape
    new_mask = mask.copy()

    for col in [0, w - 1]:  # left and right edges
        edge_column = new_mask[:, col]
        edge_indices = np.where(edge_column > 0)[0]

        if len(edge_indices) == 0:
            continue  # no edge content

        top = edge_indices[0]
        bottom = edge_indices[-1]

        # Draw a vertical line connecting top to bottom on the edge
        new_mask[top:bottom + 1, col] = 1

        # Now, find the closest internal mask pixel to the top and bottom
        non_edge_mask = new_mask.copy()
        non_edge_mask[:, col] = 0  # remove edge column

        non_zero_coords = np.column_stack(np.where(non_edge_mask > 0))
        if len(non_zero_coords) == 0:
            continue  # no internal mask to connect to

        # Top connection
        top_point = np.array([top, col])
        distances_top = np.linalg.norm(non_zero_coords - top_point, axis=1)
        closest_top = non_zero_coords[np.argmin(distances_top)]
        cv2.line(new_mask, (col, top), (closest_top[1], closest_top[0]), 1, 1)

        # Bottom connection
        bottom_point = np.array([bottom, col])
        distances_bottom = np.linalg.norm(non_zero_coords - bottom_point, axis=1)
        closest_bottom = non_zero_coords[np.argmin(distances_bottom)]
        cv2.line(new_mask, (col, bottom), (closest_bottom[1], closest_bottom[0]), 1, 1)

    return new_mask



def extract_top_bottom_edges(mask, smoothing_factor=90000):
    """
    Given a binary mask, finds and smooths the top and bottom edges that span left to right.
    Ensures the bottom edge is always below the top edge (by at least min_gap pixels).
    Returns two lists of (x, y) coordinates for the top and bottom lines.
    """
    height, width = mask.shape
    top_y = np.full(width, np.nan)
    bottom_y = np.full(width, np.nan)

    for x in range(width):
        column = np.where(mask[:, x] > 0)[0]
        if len(column) > 0:
            top_y[x] = column[0]
            bottom_y[x] = column[-1]

    valid = ~np.isnan(top_y)
    x_valid = np.arange(width)[valid]
    top_y_valid = top_y[valid]
    bottom_y_valid = bottom_y[valid]

    if len(x_valid) < 2:
        return None, None

    # Compute average mask height across valid columns
    heights = bottom_y_valid - top_y_valid
    average_height = np.mean(heights)
    min_gap = average_height / 1.7

    # Smooth the lines
    spline_top = UnivariateSpline(x_valid, top_y_valid, s=smoothing_factor)
    spline_bottom = UnivariateSpline(x_valid, bottom_y_valid, s=smoothing_factor)

    x_smooth = np.arange(width)
    y_top_smooth = spline_top(x_smooth)
    y_bottom_smooth = spline_bottom(x_smooth)

    # Enforce: bottom always below top + min_gap
    for i in range(width):
        if y_bottom_smooth[i] < y_top_smooth[i] + min_gap:
            y_bottom_smooth[i] = y_top_smooth[i] + min_gap

    # Clip to image boundaries
    y_top_smooth = np.clip(y_top_smooth, 0, height - 1)
    y_bottom_smooth = np.clip(y_bottom_smooth, 0, height - 1)

    top_line = np.column_stack((x_smooth, y_top_smooth))
    bottom_line = np.column_stack((x_smooth, y_bottom_smooth))

    return top_line, bottom_line

def create_filled_region_between_edges(top_line, bottom_line, shape):
    """
    Create a binary mask with the region between the top and bottom lines filled.
    
    Parameters:
        top_line: np.ndarray of shape (N, 2) with (x, y_top) coordinates.
        bottom_line: np.ndarray of shape (N, 2) with (x, y_bottom) coordinates.
        shape: tuple (H, W) for the output mask shape.

    Returns:
        A binary mask (np.uint8) with filled region between the top and bottom lines.
    """
    if top_line is None or bottom_line is None:
        return np.zeros(shape, dtype=np.uint8)

    # Combine top and bottom lines into one closed polygon
    polygon = np.vstack([top_line, bottom_line[::-1]])

    # Round to integer pixel coordinates
    polygon = np.round(polygon).astype(np.int32)

    # Clip coordinates to image bounds
    polygon[:, 0] = np.clip(polygon[:, 0], 0, shape[1]-1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, shape[0]-1)

    # Create mask and fill polygon
    filled_mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(filled_mask, [polygon], 1)

    return filled_mask

# Function to predict the mask on a new frame and do post processing
def predict_mask(frame, model, img_size=(308, 380), roi_mask=None):
    target_size = (frame.shape[1], frame.shape[0])

    # Apply ROI to the frame before prediction
    if roi_mask is not None:
        frame = cv2.bitwise_and(frame, frame, mask=roi_mask)

    frame_resized = cv2.resize(frame, img_size)
    if len(frame_resized.shape) == 3 and frame_resized.shape[2] == 3:
        frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    else:
        frame_gray = frame_resized

    frame_gray = frame_gray / 255.0
    frame_input = frame_gray.reshape(1, img_size[1], img_size[0], 1)

    pred_mask = model.predict(frame_input)[0, :, :, 0]
    #the number here beside "pred_mask > " is how sure the AI has to be for a part of the frame to be a mask
    #(the higher the number the more sure the AI has to be)
    pred_mask = (pred_mask > 0.002).astype(np.uint8)
    # Resize back to original frame size: width first, then height
    
    #the following functions are a bunch of post-processing functions

    pred_mask = fill_mask_holes(pred_mask) #Filling in areas that are completely surrounded by mask
    
    pred_mask = keep_largest_component(pred_mask) #Keep only the largest connected component
    
    pred_mask = connect_vertical_edges(pred_mask) #draws a line on the right and left edge from the highest and lowest points on the edge

    pred_mask = fill_mask_holes(pred_mask)

    top_line, bottom_line = extract_top_bottom_edges(pred_mask) #finds the average edge of the top and bottom of the mask and makes them smooth
    
    pred_mask = create_filled_region_between_edges(top_line, bottom_line, pred_mask.shape) #fills in space between top and bottom edges from previous function

    pred_mask = connect_vertical_edges(pred_mask)

    pred_mask = fill_mask_holes(pred_mask)

    pred_mask_resized = cv2.resize(pred_mask, target_size, interpolation=cv2.INTER_NEAREST) #resizes mask

    if roi_mask is not None:
        pred_mask_resized = cv2.bitwise_and(pred_mask_resized, pred_mask_resized, mask=roi_mask) #crops mask to ROI
    
    return pred_mask_resized

def smooth_masks_temporally(mask_sequence, kernel_size):
    """
    Applies temporal smoothing to a sequence of binary masks using a moving average filter.
    
    Parameters:
        mask_sequence (list of np.ndarray): List of binary masks (H x W).
        kernel_size (int): Number of frames to include in smoothing window (should be odd).
    
    Returns:
        smoothed_sequence (list of np.ndarray): Temporally smoothed mask sequence.
    """
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd number.")

    num_frames = len(mask_sequence)
    height, width = mask_sequence[0].shape

    # Stack into 3D array (T, H, W)
    mask_stack = np.stack(mask_sequence).astype(np.float32)

    # Pad temporally on both sides using edge padding
    pad = kernel_size // 2
    padded_stack = np.pad(mask_stack, ((pad, pad), (0, 0), (0, 0)), mode='edge')

    # Apply moving average filter
    smoothed_sequence = []
    for i in range(num_frames):
        window = padded_stack[i:i + kernel_size]
        median_mask = np.median(window, axis=0)
        smoothed_mask = (median_mask > 0.5).astype(np.uint8)
        smoothed_sequence.append(smoothed_mask)

    return smoothed_sequence

def remove_and_interpolate_jumpy_frames(mask_sequence, threshold):
    """
    Removes frames from the mask sequence that differ significantly from their neighbors
    and replaces them with an average of neighboring masks.

    Parameters:
        mask_sequence (list of np.ndarray): List of binary masks (H x W).
        threshold (float): Fraction of pixels that must differ to count as a sudden change (0-1).

    Returns:
        cleaned_sequence (list of np.ndarray): Sequence with outlier frames replaced.
    """
    cleaned_sequence = mask_sequence.copy()
    num_frames = len(mask_sequence)
    to_replace = []

    for i in range(1, num_frames - 1):
        prev_mask = mask_sequence[i - 1]
        curr_mask = mask_sequence[i]
        next_mask = mask_sequence[i + 1]

        # Difference with previous and next
        diff_prev = np.sum(prev_mask != curr_mask) / curr_mask.size
        diff_next = np.sum(next_mask != curr_mask) / curr_mask.size

        # If either differences are above threshold, mark as outlier
        if diff_prev > threshold or diff_next > threshold:
            to_replace.append(i)

    print(f"Identified {len(to_replace)} jumpy frames to replace: {to_replace}")

    for i in to_replace:
        # Average neighboring masks
        interpolated = ((mask_sequence[i - 1].astype(np.float32) + mask_sequence[i + 1].astype(np.float32)) / 2.0)
        cleaned_sequence[i] = (interpolated > 0.5).astype(np.uint8)

    return cleaned_sequence

def set_last_frame_equal_to_second_last(mask_sequence):
    """
    Sets the last frame of the mask sequence equal to the second last frame.
    """
    if len(mask_sequence) >= 2:
        mask_sequence[-1] = mask_sequence[-2].copy()
    return mask_sequence

def interpolate_over_jumpy_segments(mask_sequence, threshold, min_segment_length=3):
    """
    Detects segments where the mask suddenly changes and remains changed, 
    then reverts. Interpolates over these segments.

    Parameters:
        mask_sequence (list of np.ndarray): Binary masks.
        threshold (float): Proportion of differing pixels to consider a change.
        min_segment_length (int): Minimum length of a segment to consider it a "jump".

    Returns:
        List of np.ndarray: Corrected mask sequence.
    """
    cleaned_sequence = mask_sequence.copy()
    num_frames = len(mask_sequence)
    
    # Step 1: Compute per-frame differences with the previous frame
    diffs = [0.0]  # First frame has no previous
    for i in range(1, num_frames):
        diff = np.sum(mask_sequence[i] != mask_sequence[i-1]) / mask_sequence[i].size
        diffs.append(diff)

    # Step 2: Find start and end of jumpy segments
    in_segment = False
    segments = []
    start = None

    for i in range(1, num_frames):
        if not in_segment and diffs[i] > threshold:
            in_segment = True
            start = i - 1  # Start from the previous stable frame
        elif in_segment and diffs[i] > threshold:
            end = i  # End when we return to stability
            segment_len = end - start - 1
            if segment_len >= min_segment_length:
                segments.append((start, end))
            in_segment = False

    # Step 3: Interpolate the segments
    print(f"Detected {len(segments)} jumpy segments to interpolate: {segments}")
    for start, end in segments:
        mask_start = mask_sequence[start].astype(np.float32)
        mask_end = mask_sequence[end].astype(np.float32)
        num_interp = end - start - 1

        for idx in range(1, num_interp + 1):
            alpha = idx / (num_interp + 1)
            interp_mask = (1 - alpha) * mask_start + alpha * mask_end
            cleaned_sequence[start + idx] = (interp_mask > 0.5).astype(np.uint8)

    return cleaned_sequence


def has_path_left_to_right(mask):
    """
    Checks if there is a connected path from the left edge to the right edge in the binary mask.
    """
    structure = np.ones((3, 3))  # 8-connectivity
    labeled_mask, num_features = label(mask, structure=structure)

    if num_features == 0:
        return False

    left_labels = np.unique(labeled_mask[:, 0][labeled_mask[:, 0] > 0])
    right_labels = np.unique(labeled_mask[:, -1][labeled_mask[:, -1] > 0])

    # Check for shared label (connection)
    return len(np.intersect1d(left_labels, right_labels)) > 0


def enforce_minimum_height(mask, min_height):
    """
    Ensures that every x-column in the mask has at least min_height pixels vertically.
    If it doesn't, it expands the mask symmetrically around the center.
    """
    h, w = mask.shape
    adjusted_mask = np.zeros_like(mask)

    for x in range(w):
        column = np.where(mask[:, x] > 0)[0]
        if column.size == 0:
            continue
        top, bottom = column[0], column[-1]
        current_height = bottom - top + 1

        if current_height >= min_height:
            adjusted_mask[top:bottom + 1, x] = 1
        else:
            # Calculate new top and bottom centered on the midpoint
            mid = (top + bottom) // 2
            half_height = int(np.ceil(min_height / 2))
            new_top = max(0, mid - half_height)
            new_bottom = min(h - 1, mid + half_height)
            adjusted_mask[new_top:new_bottom + 1, x] = 1

    return adjusted_mask


def interpolate_disconnected_masks_with_path_check(mask_sequence):
    """
    Interpolates frames that do not have a connected path from left to right in the mask.
    Ensures interpolated masks are at least as tall as the average mask height.

    Parameters:
        mask_sequence (list of np.ndarray): Binary mask frames.

    Returns:
        List of np.ndarray: Fixed sequence with interpolated frames.
    """
    num_frames = len(mask_sequence)
    is_valid = []

    # Step 1: Determine valid frames
    for mask in mask_sequence:
        touches_left = np.any(mask[:, 0] > 0)
        touches_right = np.any(mask[:, -1] > 0)
        is_valid.append(touches_left and touches_right and has_path_left_to_right(mask))

    # Step 2: Compute average mask height from valid frames
    heights = []
    for i, valid in enumerate(is_valid):
        if not valid:
            continue
        mask = mask_sequence[i]
        col_heights = []
        for x in range(mask.shape[1]):
            y_coords = np.where(mask[:, x] > 0)[0]
            if len(y_coords) > 0:
                col_heights.append(y_coords[-1] - y_coords[0] + 1)
        if col_heights:
            heights.append(np.mean(col_heights))

    avg_height = int(np.mean(heights)) if heights else 10  # Fallback if no valid masks
    print(f"Average valid mask height: {avg_height}")

    corrected_masks = mask_sequence.copy()

    print_array = []
    previous_frame = []
    next_frame = []
    # Step 3: Interpolate invalid frames
    for i in range(num_frames):
        if is_valid[i]:
            continue  # Already valid

        # Find previous valid frame
        prev_idx = i - 1
        while prev_idx >= 0 and not is_valid[prev_idx]:
            prev_idx -= 1

        # Find next valid frame
        next_idx = i + 1
        while next_idx < num_frames and not is_valid[next_idx]:
            next_idx += 1

        if prev_idx >= 0 and next_idx < num_frames:
            alpha = (i - prev_idx) / (next_idx - prev_idx)
            interp = (1 - alpha) * corrected_masks[prev_idx].astype(np.float32) + alpha * corrected_masks[next_idx].astype(np.float32)
            interpolated_mask = (interp > 0.5).astype(np.uint8)
            enforced_mask = enforce_minimum_height(interpolated_mask, avg_height)
            corrected_masks[i] = enforced_mask
            print_array.append([i,prev_idx,next_idx])
        elif prev_idx >= 0:
            corrected_masks[i] = enforce_minimum_height(corrected_masks[prev_idx].copy(), avg_height)
            previous_frame.append([i,prev_idx])
        elif next_idx < num_frames:
            corrected_masks[i] = enforce_minimum_height(corrected_masks[next_idx].copy(), avg_height)
            next_frame.append([i,next_idx])
        else:
            print(f"Frame {i} is invalid with no valid neighbors. Keeping original.")
    #prints the results
    if len(print_array) >= 1:
        frames_print = []
        frame_sec_1 = print_array[0][1]
        frame_sec_2 = print_array[0][2]
        for i in print_array:
            
            

            if i[1] != frame_sec_1 or i[2] != frame_sec_2:
                print(f"Interpolate from frame {frame_sec_1} to {frame_sec_2}")
                frame_sec_1 = i[1]
                frame_sec_2 = i[2]
                frames_print = []

            frames_print.append(i[0])

        print(f"Interpolate from frame {frame_sec_1} to {frame_sec_2}")
    elif len(next_frame) == 0 and len(previous_frame) == 0:
        print("No disconnected frames (There is always a path from right to left)")
    
    if len(previous_frame) >= 1:
        frames_print = []
        frame_sec_1 = previous_frame[0][1]
        for i in previous_frame:

            if i[1] != frame_sec_1:
                print(f"Replace frames: {frames_print} with previous valid frame from {frame_sec_1}")
                frame_sec_1 = i[1]
                frames_print = []

            frames_print.append(i[0])

        print(f"Replace frames: {frames_print} with previous valid frame from {frame_sec_1}")
    if len(next_frame) >= 1:
        frames_print = []
        frame_sec_1 = next_frame[0][1]
        for i in next_frame:

            if i[1] != frame_sec_1:
                print(f"Replace frames: {frames_print} with next valid frame from {frame_sec_1}")
                frame_sec_1 = i[1]
                frames_print = []

            frames_print.append(i[0])

        print(f"Replace frames: {frames_print} with next valid frame from {frame_sec_1}")

    return corrected_masks

def load_mat_and_predict(mat_path, model, img_size=(308, 380), key='b_mode_f', roi_mask=None):
    data = loadmat(mat_path)
    
    # Check key exists
    if key not in data:
        raise KeyError(f"Key '{key}' not found in {mat_path}. Keys available: {list(data.keys())}")

    frames = data[key]
    print(f"Loaded {frames.shape[0]} frames from {mat_path}")

    all_predicted_masks = []

    #runs all frames through AI and some post processing
    for i in range(frames.shape[0]):
        
        all_predicted_masks.append(predict_mask(frames[i], model, img_size, roi_mask))

    return all_predicted_masks

# Load frames to let user draw ROI
preview_data = loadmat(f'{image_dir}\\{file_name}')
preview_frame = preview_data["b_mode_f"]  # First frame to draw ROI


viewer = napari.Viewer()
viewer.add_image(preview_frame, colormap='inferno', name='Preview Frame')
viewer.add_shapes(name='ROI', shape_type='rectangle', edge_color='cyan', face_color='cyan', opacity=0.4)

roi_mask_global = None  # to store the ROI mask
roi_box_global = None   # to store the box coordinates

@magicgui(call_button="Confirm ROI")
def confirm_roi():
    global roi_mask_global, roi_box_global

    shapes_layer = viewer.layers['ROI']
    if len(shapes_layer.data) == 0:
        print("Please draw a rectangle before confirming if you want a ROI. If not just close napari.")
        return

    # Convert from world to image coordinates
    rect_world = shapes_layer.data[0]
    rect_data = shapes_layer.world_to_data(rect_world)

    y_coords = [pt[0] for pt in rect_world]  # Note: pt[0] = y, pt[1] = x

    y1 = int(np.floor(np.min(y_coords)))
    y2 = int(np.ceil(np.max(y_coords)))

    h, w = preview_frame[0].shape

    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 1, min(y2, h))

    x1, x2 = 0, w
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    roi_mask[y1:y2, x1:x2] = 1

    roi_mask_global = roi_mask
    roi_box_global = (x1, y1, x2, y2)

    print(f"ROI set from width ({x1} to {x2}) and height ({y1} to {y2}).")
    
    QTimer.singleShot(100, viewer.close)  # Delay close to avoid button crash


viewer.window.add_dock_widget(confirm_roi, area='right')
napari.run()

# Check if ROI is being used
if roi_mask_global is None:
    predicted_masks = load_mat_and_predict(f'{image_dir}\\{file_name}', model)
else:
    #loads all frames
    predicted_masks = load_mat_and_predict(f'{image_dir}\\{file_name}', model, roi_mask=roi_mask_global)

# Smooth the mask sequence across time
predicted_masks = smooth_masks_temporally(predicted_masks, 41)

# Set the last frame equal to the second last
predicted_masks = set_last_frame_equal_to_second_last(predicted_masks)

predicted_masks = [connect_vertical_edges(mask) for mask in predicted_masks]

predicted_masks = [fill_mask_holes(mask) for mask in predicted_masks]

#predicted_masks = interpolate_over_jumpy_segments(predicted_masks, threshold=0.01)

# # Remove sudden jumpy frames
predicted_masks = remove_and_interpolate_jumpy_frames(predicted_masks, threshold=0.015)

predicted_masks = interpolate_disconnected_masks_with_path_check(predicted_masks)

#Smooth the mask sequence across time
predicted_masks = smooth_masks_temporally(predicted_masks, 41)

mask_array = np.array(predicted_masks)
print("Mask array:",mask_array.shape)
# colormap
cfi_colors = plt.cm.inferno

b_mode_f = loadmat(f'{image_dir}\\{file_name}')
b_mode_f = b_mode_f["b_mode_f"]
print("bmode array:",b_mode_f.shape)

print("Unique values in mask array (should be [0 1]):", np.unique(mask_array))

print("Model used:",model_name)

print("File name",file_name)

viewer = napari.Viewer()
viewer.add_image(b_mode_f, colormap='inferno', name='Filtered B-Mode')#image
viewer.add_labels(mask_array, name='Segmentation Mask')
# Run the viewer
napari.run()

#code to export mask
file_name, other = file_name.split(")")
file_name = file_name+")_masks.mat"
saved_files = get_files(save_dir)
if file_name not in saved_files:
    savemat(os.path.join(save_dir, file_name), {'mask_array': mask_array})
    print("Exported:",file_name,"to",save_dir)
else:
    file_version = 2
    file_name, other = file_name.split(".")
    while ((file_name+"_v"+str(file_version)+".mat") in saved_files):
        file_version += 1
    file_name = file_name+"_v"+str(file_version)+".mat"
    savemat(os.path.join(save_dir, file_name), {'mask_array': mask_array})
    print("Exported:",file_name,"to",save_dir)
