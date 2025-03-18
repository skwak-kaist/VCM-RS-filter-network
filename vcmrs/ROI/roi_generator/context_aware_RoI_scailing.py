import torch
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# Set the device to GPU or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Depth estimation function using the MiDaS model
def estimate_depth(image):
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(device)  # Move the model to the device
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
    input_batch = midas_transforms(image).to(device)  # Move the image to the device
    
    with torch.no_grad():
        prediction = midas(input_batch)
        depth_map = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    return depth_map.cpu().numpy()  # Move the result back to CPU and return it

# Configuration setup and model loading function
def model_cfg_set(model_pth):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_pth
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80  # Number of classes in the COCO dataset
    cfg.TEST.DETECTIONS_PER_IMAGE = 200
    cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    return cfg

# Function to filter only person instances
def filter_person_instances(outputs):
    person_indices = (outputs["instances"].pred_classes == 0)  # 0 is the person class in COCO
    return outputs["instances"][person_indices]

# Convert all non-object areas to gray and draw scaled bounding boxes
def draw_person_with_gray_background_and_bboxes(image, instances, scaled_rois, include_box_area):
    gray_image = np.full_like(image, 128)  # Create a gray image
    mask = np.zeros_like(image)

    for original_box, scaled_box in zip(instances.pred_boxes.tensor.tolist(), scaled_rois):
        x1, y1, x2, y2 = map(int, original_box)  # Original object's bounding box
        scaled_x1, scaled_y1, scaled_x2, scaled_y2 = map(int, scaled_box)  # Scaled bounding box

        if include_box_area:
            # Keep both the object region and the expanded scaled region
            mask[scaled_y1:scaled_y2, scaled_x1:scaled_x2] = image[scaled_y1:scaled_y2, scaled_x1:scaled_x2]
        else:
            # Copy only the object region from the original image
            mask[y1:y2, x1:x2] = image[y1:y2, x1:x2]

        # Draw the scaled bounding box
        cv2.rectangle(mask, (scaled_x1, scaled_y1), (scaled_x2, scaled_y2), (0, 255, 0), 2)  # Green box

    combined_image = np.where(mask.sum(axis=-1, keepdims=True) > 0, mask, gray_image)  # Show objects in color, others in gray
    return combined_image

# Scaling based on RoI size (method 1)
# Adaptive scaling based on RoI size
def scale_rois_by_size(rois_list, base_scale_factor=1.0, min_size=(30, 30), max_size=None, image_size=None):
    if len(rois_list) == 0:
        return rois_list  # Return immediately if no RoIs
    
    height = image_size[0]
    width = image_size[1]
    if max_size is None:
        max_size = (width, height)
    
    scaled_rois = []
    scaled_rois_list = []
    base_area = (width * height) / 1000 
    
    for rois in rois_list:
        for roi in rois:
            x1, y1, x2, y2 = roi
            roi_width = x2 - x1
            roi_height = y2 - y1
            # Calculate RoI area
            roi_area = roi_width * roi_height
           # Calculate adaptive scale factor
            if roi_area <= 0:
                scale_factor = 1
            else:
                scale_factor = base_scale_factor + (base_area / roi_area)
            scale_factor = max(base_scale_factor, min(scale_factor, 2))

        # Calculate new RoI size
            new_width = roi_width * scale_factor
            new_height = roi_height * scale_factor

        # Apply minimum and maximum size
            new_width = max(min_size[0], min(new_width, max_size[0]))
            new_height = max(min_size[1], min(new_height, max_size[1]))

        # Recalculate RoI coordinates based on center
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            new_x1 = int(max(0, center_x - new_width // 2))
            new_y1 = int(max(0, center_y - new_height // 2))
            new_x2 = int(min(width, center_x + new_width // 2))
            new_y2 = int(min(height, center_y + new_height // 2))
            scaled_rois.append([new_x1, new_y1, new_x2, new_y2])
        scaled_rois_list.append(scaled_rois)
        scaled_rois = []
    return scaled_rois_list

# Scaling based on object importance and depth information (method 2)
def scale_rois_by_importance(rois, depth_map, include_box_area):
    height, width = depth_map.shape  # Use depth_map size to set width and height
    scaled_rois = []

    for roi in rois:
        x1, y1, x2, y2 = map(int, roi)  # Convert coordinates to integers
        roi_width = x2 - x1
        roi_height = y2 - y1

        # Set higher scale factor for more important objects (using depth value)
        roi_depth = np.mean(depth_map[y1:y2, x1:x2])
        scale_factor = 1 / (roi_depth + 1e-6)  # Scaling based on depth

        # Limit the scale factor to avoid too small or too large adjustments
        scale_factor = max(1.2, min(scale_factor, 3.0))  # Set lower limit 1.2, upper limit 3.0

        # Process based on whether confirmed regions are included
        if include_box_area:
            new_width = int(roi_width * scale_factor)
            new_height = int(roi_height * scale_factor)
        else:
            # If include_box_area is False, keep original size
            new_width = roi_width
            new_height = roi_height

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        new_x1 = max(0, center_x - new_width // 2)
        new_y1 = max(0, center_y - new_height // 2)
        new_x2 = min(width, center_x + new_width // 2)
        new_y2 = min(height, center_y + new_height // 2)

        scaled_rois.append([new_x1, new_y1, new_x2, new_y2])

    return scaled_rois

# Scaling based on surrounding context (method 3)
def scale_rois_by_context(rois, image_size, min_neighbors=1, max_neighbors=5):
    """
    Adjust RoI size based on the number of nearby objects.
    
    :param rois: List of RoIs
    :param image_size: Image size (height, width)
    :param min_neighbors: Minimum number of nearby objects (default: 1)
    :param max_neighbors: Maximum number of nearby objects (default: 5)
    """
    height = image_size[0]
    width = image_size[1]
    scaled_rois = []

    # Process all objects
    for i, roi in enumerate(rois):
        x1, y1, x2, y2 = map(int, roi)
        roi_width = x2 - x1
        roi_height = y2 - y1
        
        # Calculate the center of the current RoI
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Calculate the number of nearby objects
        neighbors = 0
        for j, other_roi in enumerate(rois):
            if i != j:
                other_x1, other_y1, other_x2, other_y2 = map(int, other_roi)
                other_center_x = (other_x1 + other_x2) // 2
                other_center_y = (other_y1 + other_y2) // 2
                
                # Calculate distance between centers (e.g., Euclidean distance)
                distance = np.sqrt((center_x - other_center_x)**2 + (center_y - other_center_y)**2)
                
                # Check if it is a nearby object (arbitrarily considered within 100 pixels)
                if distance < 100:
                    neighbors += 1
        
        # Adjust scale factor based on the number of nearby objects
        if neighbors >= max_neighbors:
            scale_factor = 1.5  # Expand more if there are many nearby objects
        elif neighbors >= min_neighbors:
            scale_factor = 1.3  # Expand slightly if there are some nearby objects
        else:
            scale_factor = 1.0  # Keep size unchanged if there are few or no nearby objects
        
        # Calculate the scaled RoI size
        new_width = int(roi_width * scale_factor)
        new_height = int(roi_height * scale_factor)
        
        # Calculate new RoI coordinates
        new_x1 = max(0, center_x - new_width // 2)
        new_y1 = max(0, center_y - new_height // 2)
        new_x2 = min(width, center_x + new_width // 2)
        new_y2 = min(height, center_y + new_height // 2)
        
        scaled_rois.append([new_x1, new_y1, new_x2, new_y2])

    return scaled_rois

# Perform scaling based on the selected method
def scale_rois(rois, scaling_method, depth_map=None, image_size=None, include_box_area=False):
    if scaling_method == 1:
        return scale_rois_by_size(rois, image_size=image_size)  # Pass image_size
    elif scaling_method == 2:
        if depth_map is not None:
            return scale_rois_by_importance(rois, depth_map, include_box_area)  # Pass include_box_area
        else:
            raise ValueError("Depth map is required for importance-based scaling")
    elif scaling_method == 3:
        if image_size is not None:
            return scale_rois_by_context(rois, image_size)  # Pass image_size
        else:
            raise ValueError("Image size is required for context-based scaling")
    else:
        raise ValueError("Invalid scaling method selected")

