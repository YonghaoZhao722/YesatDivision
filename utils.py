import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import exposure
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

def preprocess_image(image, method="Basic"):
    """
    Preprocess the input image using different methods.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image to preprocess
    method : str
        Preprocessing method to apply
        
    Returns:
    --------
    processed_image : numpy.ndarray
        Preprocessed image
    """
    # Make a copy to avoid modifying the original
    processed_image = image.copy()
    
    # Ensure image is in the right format for processing
    if processed_image.dtype != np.uint8:
        if processed_image.dtype == np.uint16:
            # Scale 16-bit to 8-bit
            processed_image = (processed_image / 256).astype(np.uint8)
        elif processed_image.dtype == np.float32 or processed_image.dtype == np.float64:
            if processed_image.max() <= 1.0:
                processed_image = (processed_image * 255).astype(np.uint8)
            else:
                processed_image = processed_image.astype(np.uint8)
        else:
            processed_image = processed_image.astype(np.uint8)
    
    # Convert to float32 for processing, scaled to 0-1
    processed_image_float = processed_image.astype(np.float32) / 255.0
    
    # Apply different preprocessing methods
    if method == "Basic":
        # Basic normalization
        if processed_image_float.min() != processed_image_float.max():
            processed_image_float = (processed_image_float - processed_image_float.min()) / (processed_image_float.max() - processed_image_float.min())
        # No change needed if min==max (uniform image)
        
    elif method == "Contrast Enhancement":
        # Adaptive histogram equalization for better contrast
        processed_image_float = exposure.equalize_adapthist(processed_image_float)
        
    elif method == "Noise Reduction":
        # First convert back to uint8 for GaussianBlur
        temp_img = (processed_image_float * 255).astype(np.uint8)
        # Apply gaussian blur
        temp_img = cv2.GaussianBlur(temp_img, (5, 5), 0)
        # Back to float
        processed_image_float = temp_img.astype(np.float32) / 255.0
        # Then contrast enhancement
        processed_image_float = exposure.equalize_adapthist(processed_image_float)
    
    # Convert back to 0-255 range for return
    processed_image = (processed_image_float * 255).astype(np.uint8)
    
    return processed_image

def create_visualization(original_image, mask, division_events, labeled_cells):
    """
    Create a visualization of the detected cell division events.
    
    Parameters:
    -----------
    original_image : numpy.ndarray
        Original phase contrast image
    mask : numpy.ndarray
        Binary segmentation mask
    division_events : list
        List of division events detected
    labeled_cells : numpy.ndarray
        Labeled image with unique ID for each cell
        
    Returns:
    --------
    visualization : numpy.ndarray
        Visualization image showing detected division events
    """
    # Create a color visualization
    if len(original_image.shape) == 2:
        # Convert grayscale to RGB
        vis_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    else:
        vis_image = original_image.copy()
    
    # Make sure image is in the right format for display
    if vis_image.dtype != np.uint8:
        vis_image = (vis_image * 255).astype(np.uint8)
    
    # Create a colormap for labeled cells
    cmap = plt.cm.get_cmap('tab20', np.max(labeled_cells) + 1)
    
    # Create an overlay for the segmentation
    overlay = np.zeros_like(vis_image)
    
    # Fill overlay with colors for each cell
    for label_id in range(1, np.max(labeled_cells) + 1):
        cell_mask = labeled_cells == label_id
        color = np.array(cmap(label_id)[:3]) * 255
        for i in range(3):
            overlay[cell_mask, i] = color[i]
    
    # Add segmentation as semi-transparent overlay
    alpha = 0.3
    vis_image = cv2.addWeighted(overlay, alpha, vis_image, 1 - alpha, 0)
    
    # Draw division events
    for event in division_events:
        mother = event['mother_cell']
        daughter = event['daughter_cell']
        
        # Draw centers of cells
        mother_center = (int(mother['center'][0]), int(mother['center'][1]))
        daughter_center = (int(daughter['center'][0]), int(daughter['center'][1]))
        
        # Draw a line connecting the cells
        cv2.line(vis_image, mother_center, daughter_center, (255, 255, 0), 2)
        
        # Draw circles for mother (red) and daughter (green) cells
        cv2.circle(vis_image, mother_center, 10, (255, 0, 0), 2)  # Red for mother
        cv2.circle(vis_image, daughter_center, 8, (0, 255, 0), 2)  # Green for daughter
        
        # Add labels
        cv2.putText(vis_image, "M", (mother_center[0] + 15, mother_center[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(vis_image, "D", (daughter_center[0] + 15, daughter_center[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add confidence score
        confidence_pos = ((mother_center[0] + daughter_center[0]) // 2,
                          (mother_center[1] + daughter_center[1]) // 2 - 15)
        cv2.putText(vis_image, f"{event['confidence']:.2f}", confidence_pos, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return vis_image

def format_results(division_events, labeled_cells):
    """
    Format the division events results for display.
    
    Parameters:
    -----------
    division_events : list
        List of division events detected
    labeled_cells : numpy.ndarray
        Labeled image with unique ID for each cell
        
    Returns:
    --------
    formatted_results : pandas.DataFrame
        Formatted results as a pandas DataFrame
    """
    if not division_events:
        return pd.DataFrame()
    
    # Create results table
    results = []
    
    for i, event in enumerate(division_events):
        mother = event['mother_cell']
        daughter = event['daughter_cell']
        
        result = {
            "Event #": i + 1,
            "Mother Cell ID": mother['label'],
            "Daughter Cell ID": daughter['label'],
            "Distance (px)": f"{event['distance']:.2f}",
            "Size Ratio": f"{event['size_ratio']:.2f}",
            "Mother Area": f"{mother['area']}",
            "Daughter Area": f"{daughter['area']}",
            "Confidence": f"{event['confidence']:.2f}"
        }
        
        results.append(result)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    return df

def measure_cell_properties(image, mask, cell_id):
    """
    Measure additional properties of a cell.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Original image
    mask : numpy.ndarray
        Segmentation mask
    cell_id : int
        ID of the cell to analyze
        
    Returns:
    --------
    properties : dict
        Dictionary of cell properties
    """
    cell_mask = mask == cell_id
    
    # Measure intensity
    if len(image.shape) == 3:  # RGB image
        intensity = np.mean(image[cell_mask], axis=0)
    else:
        intensity = np.mean(image[cell_mask])
    
    # Measure shape features
    contours, _ = cv2.findContours(cell_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return {
            'intensity': intensity,
            'area': 0,
            'perimeter': 0,
            'circularity': 0
        }
    
    # Get the largest contour
    contour = max(contours, key=cv2.contourArea)
    
    # Calculate basic shape properties
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    # Calculate circularity (1 is perfect circle)
    circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
    
    return {
        'intensity': intensity,
        'area': area,
        'perimeter': perimeter,
        'circularity': circularity
    }
