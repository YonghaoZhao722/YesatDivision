import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import exposure, morphology
from skimage.util import img_as_float, img_as_ubyte
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
    
    # Make sure all values are between 0 and 1 before scaling to 255
    processed_image_float = np.clip(processed_image_float, 0.0, 1.0)
    
    # Convert back to 0-255 range for return
    processed_image = (processed_image_float * 255).astype(np.uint8)
    
    return processed_image

def create_visualization(original_image, mask, division_events, labeled_cells, overlay_opacity=0.3):
    """
    Create a visualization of the detected cell division events, similar to ImageJ/Fiji's overlay feature.
    
    Parameters:
    -----------
    original_image : numpy.ndarray
        Original phase contrast image (or auto-contrasted version)
    mask : numpy.ndarray
        Binary segmentation mask
    division_events : list
        List of division events detected
    labeled_cells : numpy.ndarray
        Labeled image with unique ID for each cell
    overlay_opacity : float
        Opacity of the segmentation mask overlay (0.0-1.0, default: 0.3 or 30%)
        
    Returns:
    --------
    visualization : numpy.ndarray
        Visualization image showing detected division events with Fiji-like overlay
    """
    # Create a color visualization
    if len(original_image.shape) == 2:
        # Convert grayscale to RGB
        vis_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    else:
        vis_image = original_image.copy()
    
    # Make sure image is in the right format for display
    if vis_image.dtype != np.uint8:
        if vis_image.max() <= 1.0:
            vis_image = (vis_image * 255).astype(np.uint8)
        else:
            vis_image = np.clip(vis_image, 0, 255).astype(np.uint8)
    
    # Generate a Fiji-like colored version of the mask using a colormap
    # Apply 'fire' LUT (similar to Fiji)
    cmap_mask = plt.cm.get_cmap('hot')
    
    # First convert binary mask to a more visually interesting representation
    mask_for_vis = mask.copy()
    if mask_for_vis.dtype == np.uint8:
        # Convert to float for better gradient
        mask_for_vis = mask_for_vis.astype(np.float32) / 255.0
    
    # Apply the colormap
    colored_mask = cmap_mask(mask_for_vis)
    colored_mask = (colored_mask[:, :, :3] * 255).astype(np.uint8)
    
    # Create a Fiji-like overlay with distinct colors for each cell
    # This simulates the "glasbey" LUT in Fiji which assigns distinct colors to labeled objects
    
    # Create colored overlay image (transparent where no cells exist)
    overlay_img = np.zeros_like(vis_image)
    
    # Number of unique cell IDs
    num_cells = np.max(labeled_cells) if np.max(labeled_cells) > 0 else 0
    
    # Use a Fiji-like glasbey colormap with distinct bright colors
    # For reproducibility, we'll use a fixed color palette similar to Fiji
    glasbey_colors = [
        (255, 0, 0),     # Red
        (0, 255, 0),     # Green
        (0, 0, 255),     # Blue
        (255, 255, 0),   # Yellow
        (255, 0, 255),   # Magenta
        (0, 255, 255),   # Cyan
        (255, 128, 0),   # Orange
        (128, 0, 255),   # Purple
        (0, 255, 128),   # Mint
        (255, 128, 128), # Pink
        (128, 255, 0),   # Lime
        (0, 128, 255),   # Sky blue
        (255, 0, 128),   # Rose
        (128, 0, 128),   # Violet
        (128, 128, 0),   # Olive
        (0, 128, 128),   # Teal
        (255, 128, 255), # Light pink
        (128, 255, 128), # Light green
        (128, 128, 255), # Light blue
        (255, 255, 128), # Light yellow
    ] 
    
    # Draw colored overlay for each cell
    for label_id in range(1, num_cells + 1):
        # Get the mask for this specific cell
        cell_mask = labeled_cells == label_id
        
        # Skip empty masks
        if not np.any(cell_mask):
            continue
            
        # Get color from the glasbey color map
        color_idx = (label_id - 1) % len(glasbey_colors)
        cell_color = glasbey_colors[color_idx]
        
        # Find the cell contour for better edge highlighting
        contours, _ = cv2.findContours(
            cell_mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Fill the cell area with color
        for i in range(3):
            overlay_img[:, :, i] = np.where(cell_mask, cell_color[i], overlay_img[:, :, i])
        
        # Draw a slightly thicker outline around each cell
        cv2.drawContours(overlay_img, contours, -1, cell_color, 2)
    
    # Apply the overlay with user-specified opacity
    vis_image = cv2.addWeighted(overlay_img, overlay_opacity, vis_image, 1.0, 0)
    
    # Draw division events with clearer markers
    for event in division_events:
        mother = event['mother_cell']
        daughter = event['daughter_cell']
        
        # Draw centers of cells
        mother_center = (int(mother['center'][0]), int(mother['center'][1]))
        daughter_center = (int(daughter['center'][0]), int(daughter['center'][1]))
        
        # Draw a thicker line connecting the cells with yellow color
        cv2.line(vis_image, mother_center, daughter_center, (255, 255, 0), 3)
        
        # Draw circles for mother (red) and daughter (green) cells with more visibility
        cv2.circle(vis_image, mother_center, 10, (255, 0, 0), 3)  # Red for mother
        cv2.circle(vis_image, daughter_center, 8, (0, 255, 0), 3)  # Green for daughter
        
        # Add labels with better contrast for visibility
        # Add white background to text for better readability
        text_size = cv2.getTextSize("M", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        
        # Mother cell label
        cv2.putText(vis_image, "M", (mother_center[0] + 15, mother_center[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)  # Black outline
        cv2.putText(vis_image, "M", (mother_center[0] + 15, mother_center[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  # White text
        
        # Daughter cell label
        cv2.putText(vis_image, "D", (daughter_center[0] + 15, daughter_center[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)  # Black outline
        cv2.putText(vis_image, "D", (daughter_center[0] + 15, daughter_center[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  # White text
        
        # Add confidence score with better visibility
        confidence_pos = ((mother_center[0] + daughter_center[0]) // 2,
                          (mother_center[1] + daughter_center[1]) // 2 - 15)
        
        # Add black outline to text for better readability
        cv2.putText(vis_image, f"{event['confidence']:.2f}", confidence_pos, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)  # Black outline
        cv2.putText(vis_image, f"{event['confidence']:.2f}", confidence_pos, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)  # White text
    
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

def auto_contrast(image, clip_percent=0.5, gamma=1.0):
    """
    Apply auto contrast enhancement similar to Fiji's Auto Brightness/Contrast function.
    This implementation closely mimics Fiji's algorithm for better visualization of microscopy images.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image to enhance
    clip_percent : float
        Percentage of pixels to clip from histogram (0-100)
    gamma : float
        Gamma correction value
    
    Returns:
    --------
    enhanced_image : numpy.ndarray
        Contrast-enhanced image ready for display
    """
    # Make a copy to avoid modifying the original
    img_copy = image.copy()
    
    # For 16-bit TIF images, we need special handling
    if img_copy.dtype == np.uint16:
        # Convert directly to float32 to preserve precision
        img_float = img_copy.astype(np.float32)
        
        # Determine actual bit depth from data (some 16-bit images only use 12 bits)
        max_possible = 65535.0
        actual_max = np.max(img_float)
        actual_range = max(actual_max, 1.0)  # Avoid div by zero
        
        # Calculate percentiles for contrast enhancement
        low_percentile = clip_percent / 2.0
        high_percentile = 100 - (clip_percent / 2.0)
        
        # Determine limits based on histogram
        p_low = np.percentile(img_float, low_percentile)
        p_high = np.percentile(img_float, high_percentile)
        
        # Apply contrast stretching
        enhanced = exposure.rescale_intensity(
            img_float, 
            in_range=(p_low, p_high), 
            out_range=(0, 255)  # Always output in 8-bit range for display
        )
        
        # Apply gamma correction if needed
        if gamma != 1.0:
            # Normalize to 0-1 for gamma
            enhanced_norm = enhanced / 255.0
            enhanced = np.power(enhanced_norm, gamma) * 255.0
            
        # Return as uint8 for display
        return np.clip(enhanced, 0, 255).astype(np.uint8)
        
    else:
        # For 8-bit images or others
        if img_copy.dtype != np.float32 and img_copy.dtype != np.float64:
            # Convert to float for processing
            img_float = img_as_float(img_copy)
        else:
            img_float = img_copy.copy()
            
        # Calculate percentiles
        low_percentile = clip_percent / 2.0
        high_percentile = 100 - (clip_percent / 2.0)
        
        if len(img_float.shape) > 2:  # Color image
            enhanced = np.zeros_like(img_float)
            for i in range(img_float.shape[2]):
                channel = img_float[..., i]
                if np.min(channel) != np.max(channel):  # Avoid division by zero
                    # Get histogram limits
                    p_low = np.percentile(channel, low_percentile)
                    p_high = np.percentile(channel, high_percentile)
                    # Fiji-like auto contrast
                    enhanced_channel = exposure.rescale_intensity(
                        channel, 
                        in_range=(p_low, p_high), 
                        out_range=(0, 1.0)
                    )
                    enhanced[..., i] = enhanced_channel
                else:
                    enhanced[..., i] = channel
        else:  # Grayscale image
            if np.min(img_float) != np.max(img_float):  # Avoid division by zero
                # Get histogram limits
                p_low = np.percentile(img_float, low_percentile)
                p_high = np.percentile(img_float, high_percentile)
                # Fiji-like auto contrast
                enhanced = exposure.rescale_intensity(
                    img_float, 
                    in_range=(p_low, p_high), 
                    out_range=(0, 1.0)
                )
            else:
                enhanced = img_float
                
        # Apply gamma correction
        if gamma != 1.0:
            enhanced = np.power(enhanced, gamma)
        
        # Convert to 8-bit for display
        return img_as_ubyte(enhanced)

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
