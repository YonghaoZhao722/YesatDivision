import numpy as np
import cv2
from scipy.ndimage import label, center_of_mass, distance_transform_edt
from skimage.measure import regionprops
from skimage.feature import graycomatrix, graycoprops
import pandas as pd
from scipy.spatial.distance import pdist, squareform

class CellDivisionAnalyzer:
    """
    A class to analyze cell division events in microscopy images of yeast cells.
    """
    
    def __init__(self, distance_threshold=15, size_ratio_threshold=0.7, min_cell_size=100):
        """
        Initialize the cell division analyzer.
        
        Parameters:
        -----------
        distance_threshold : int
            Maximum distance between cells to be considered as a division event
        size_ratio_threshold : float
            Threshold for size ratio to distinguish mother and daughter cells
        min_cell_size : int
            Minimum size (in pixels) for a region to be considered a cell
        """
        self.distance_threshold = distance_threshold
        self.size_ratio_threshold = size_ratio_threshold
        self.min_cell_size = min_cell_size
    
    def analyze(self, image, mask):
        """
        Analyze the image and mask to detect cell division events.
        
        Parameters:
        -----------
        image : numpy.ndarray
            Phase contrast image of yeast cells
        mask : numpy.ndarray
            Binary segmentation mask of the cells
        
        Returns:
        --------
        division_events : list
            List of dictionaries containing information about each division event
        labeled_cells : numpy.ndarray
            Labeled image with unique ID for each cell
        """
        # Label connected components in the mask
        labeled_cells, num_cells = label(mask > 0)
        
        if num_cells < 2:
            return [], labeled_cells
        
        # Extract properties for each region
        cell_props = regionprops(labeled_cells)
        
        # Filter cells by size
        valid_cells = [prop for prop in cell_props if prop.area >= self.min_cell_size]
        
        if len(valid_cells) < 2:
            return [], labeled_cells
        
        # Store cell info
        cells = []
        for prop in valid_cells:
            # Get cell center
            center_y, center_x = prop.centroid
            
            # Calculate average intensity in the original image
            cell_mask = labeled_cells == prop.label
            cell_intensity = np.mean(image[cell_mask])
            
            # Calculate roundness (1.0 means perfect circle)
            perimeter = prop.perimeter
            roundness = (4 * np.pi * prop.area) / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Create cell data
            cell = {
                'label': prop.label,
                'center': (center_x, center_y),
                'area': prop.area,
                'perimeter': perimeter,
                'roundness': roundness,
                'intensity': cell_intensity,
                'bounding_box': prop.bbox,
                'mean_intensity': cell_intensity
            }
            cells.append(cell)
        
        # Find potential division events by calculating distances between cells
        division_events = []
        
        for i in range(len(cells)):
            for j in range(i+1, len(cells)):
                # Calculate Euclidean distance between cell centers
                distance = np.sqrt(
                    (cells[i]['center'][0] - cells[j]['center'][0]) ** 2 +
                    (cells[i]['center'][1] - cells[j]['center'][1]) ** 2
                )
                
                # Check if cells are close enough to be a division event
                if distance <= self.distance_threshold:
                    # Identify mother and daughter cells based on size
                    if cells[i]['area'] >= cells[j]['area']:
                        mother_idx, daughter_idx = i, j
                    else:
                        mother_idx, daughter_idx = j, i
                    
                    # Calculate size ratio (daughter to mother)
                    size_ratio = cells[daughter_idx]['area'] / cells[mother_idx]['area']
                    
                    # Create event data
                    event = {
                        'mother_cell': cells[mother_idx],
                        'daughter_cell': cells[daughter_idx],
                        'distance': distance,
                        'size_ratio': size_ratio,
                        'confidence': self._calculate_confidence(distance, size_ratio)
                    }
                    division_events.append(event)
        
        return division_events, labeled_cells
    
    def _calculate_confidence(self, distance, size_ratio):
        """
        Calculate a confidence score for a division event based on distance and size ratio.
        
        Parameters:
        -----------
        distance : float
            Distance between the two cells
        size_ratio : float
            Ratio of daughter cell size to mother cell size
        
        Returns:
        --------
        confidence : float
            Confidence score between 0 and 1
        """
        # Distance factor: lower distance = higher confidence
        distance_factor = max(0, 1 - distance / self.distance_threshold)
        
        # Size ratio factor: size_ratio close to threshold = higher confidence
        # Mother cells should be larger than daughter cells
        ideal_ratio = self.size_ratio_threshold / 2  # Ideal ratio is half of the threshold
        size_ratio_factor = max(0, 1 - abs(size_ratio - ideal_ratio) / ideal_ratio)
        
        # Combine factors with different weights
        confidence = 0.7 * distance_factor + 0.3 * size_ratio_factor
        
        return min(1.0, max(0.0, confidence))
    
    def detect_cell_wall(self, image, mask, labeled_cells, cell_id):
        """
        Detect the cell wall for a given cell.
        
        Parameters:
        -----------
        image : numpy.ndarray
            Original phase contrast image
        mask : numpy.ndarray
            Binary segmentation mask
        labeled_cells : numpy.ndarray
            Labeled image with unique ID for each cell
        cell_id : int
            ID of the cell to analyze
            
        Returns:
        --------
        cell_wall : numpy.ndarray
            Binary image highlighting the cell wall
        """
        # Create a binary mask for the specific cell
        cell_mask = labeled_cells == cell_id
        
        # Calculate distance transform
        distance = distance_transform_edt(cell_mask)
        
        # Threshold distance to get inner region
        inner_region = distance > 2  # Adjust this value based on cell wall thickness
        
        # Cell wall is the difference between the cell mask and the inner region
        cell_wall = cell_mask & ~inner_region
        
        return cell_wall
    
    def calculate_touching_area(self, labeled_cells, cell1_id, cell2_id):
        """
        Calculate the area where two cells touch each other.
        
        Parameters:
        -----------
        labeled_cells : numpy.ndarray
            Labeled image with unique ID for each cell
        cell1_id : int
            ID of the first cell
        cell2_id : int
            ID of the second cell
            
        Returns:
        --------
        touch_area : int
            Number of pixels in the touching area
        """
        # Create binary masks for each cell
        cell1_mask = labeled_cells == cell1_id
        cell2_mask = labeled_cells == cell2_id
        
        # Dilate both masks
        kernel = np.ones((3, 3), np.uint8)
        cell1_dilated = cv2.dilate(cell1_mask.astype(np.uint8), kernel, iterations=1)
        cell2_dilated = cv2.dilate(cell2_mask.astype(np.uint8), kernel, iterations=1)
        
        # Find intersection of dilated areas and remove original cells
        touch_area = (cell1_dilated & cell2_dilated) & ~(cell1_mask | cell2_mask)
        
        return np.sum(touch_area)
        
    def _extract_cell_features(self, image, labeled_cells, cell_prop):
        """
        Extract advanced features for a cell.
        
        Parameters:
        -----------
        image : numpy.ndarray
            Original phase contrast image
        labeled_cells : numpy.ndarray
            Labeled image with unique ID for each cell
        cell_prop : RegionProp
            Properties of the cell region
            
        Returns:
        --------
        features : dict
            Dictionary of cell features
        """
        # Create mask for this cell
        cell_mask = labeled_cells == cell_prop.label
        
        # Extract basic shape features
        area = cell_prop.area
        perimeter = cell_prop.perimeter
        eccentricity = cell_prop.eccentricity
        
        # Calculate roundness (circularity)
        roundness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Calculate mean and standard deviation of intensity
        if len(image.shape) == 3:  # RGB
            intensity_mean = np.mean(image[cell_mask], axis=0)
            intensity_std = np.std(image[cell_mask], axis=0)
            # Average across channels
            intensity_mean = np.mean(intensity_mean)
            intensity_std = np.mean(intensity_std)
        else:  # Grayscale
            intensity_mean = np.mean(image[cell_mask])
            intensity_std = np.std(image[cell_mask])
        
        # Detect cell wall
        cell_wall = self.detect_cell_wall(image, cell_mask, labeled_cells, cell_prop.label)
        wall_thickness = np.sum(cell_wall) / perimeter if perimeter > 0 else 0
        
        # Extract bounding box and calculate aspect ratio
        min_row, min_col, max_row, max_col = cell_prop.bbox
        width = max_col - min_col
        height = max_row - min_row
        aspect_ratio = width / height if height > 0 else 0
        
        # Calculate texture features if possible (requires grayscale image)
        try:
            # Extract the cell region
            y_min, x_min, y_max, x_max = cell_prop.bbox
            cell_region = image[y_min:y_max, x_min:x_max].copy()
            
            # Apply cell mask to get only cell pixels
            mask_region = cell_mask[y_min:y_max, x_min:x_max]
            if not np.any(mask_region):
                # No valid pixels in this region
                contrast = homogeneity = energy = correlation = 0
            else:
                # Normalize to 0-255 range for GLCM
                if cell_region.max() > 0:
                    cell_region = ((cell_region - cell_region.min()) / (cell_region.max() - cell_region.min()) * 255).astype(np.uint8)
                
                # Set background to 0
                cell_region[~mask_region] = 0
                
                # Calculate GLCM
                if np.any(cell_region):
                    glcm = graycomatrix(cell_region, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 256, symmetric=True, normed=True)
                    # Extract features
                    contrast = np.mean(graycoprops(glcm, 'contrast')[0])
                    homogeneity = np.mean(graycoprops(glcm, 'homogeneity')[0])
                    energy = np.mean(graycoprops(glcm, 'energy')[0])
                    correlation = np.mean(graycoprops(glcm, 'correlation')[0])
                else:
                    contrast = homogeneity = energy = correlation = 0
        except Exception as e:
            # If texture features fail, use defaults
            contrast = homogeneity = energy = correlation = 0
        
        # Return the feature dictionary
        return {
            'area': area,
            'perimeter': perimeter,
            'roundness': roundness,
            'eccentricity': eccentricity,
            'intensity_mean': intensity_mean,
            'intensity_std': intensity_std,
            'wall_thickness': wall_thickness,
            'aspect_ratio': aspect_ratio,
            'contrast': contrast,
            'homogeneity': homogeneity,
            'energy': energy,
            'correlation': correlation
        }
        
    def analyze_with_ml(self, image, mask, confidence_threshold=0.3):
        """
        Analyze the image and mask to detect cell division events using machine learning features.
        
        Parameters:
        -----------
        image : numpy.ndarray
            Phase contrast image of yeast cells
        mask : numpy.ndarray
            Binary segmentation mask of the cells (ground truth)
        confidence_threshold : float
            Minimum confidence score to consider a cell division event valid
            
        Returns:
        --------
        division_events : list
            List of dictionaries containing information about each division event
        labeled_cells : numpy.ndarray
            Labeled image with unique ID for each cell
        """
        # Make sure the mask is binary
        mask_binary = (mask > 0).astype(np.uint8) * 255
        
        # Store a copy of the original mask for visualization
        original_mask = mask_binary.copy()
        
        # First, use the connected components from the original ground truth mask
        # This preserves the exact cell shapes from the uploaded mask
        labeled_original, _ = cv2.connectedComponents(mask_binary)
        
        # Also apply watershed to help separate touching cells where needed
        # This helps in detecting buds that might be connected in the binary mask
        
        # First find sure background
        sure_bg = cv2.dilate(mask_binary, np.ones((3,3), np.uint8), iterations=1)
        
        # Finding sure foreground area using distance transform
        dist_transform = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.3*dist_transform.max(), 255, 0)  # Lower threshold to capture more cells
        sure_fg = sure_fg.astype(np.uint8)
        
        # Finding unknown region
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Label the foreground objects
        _, markers = cv2.connectedComponents(sure_fg)
        
        # Add one to all labels so that background is 1 instead of 0
        markers = markers + 1
        
        # Mark the unknown region with 0
        markers[unknown == 255] = 0
        
        # Apply watershed
        if len(image.shape) == 2:
            # Convert to 3-channel for watershed
            image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_color = image.copy()
            
        # Apply watershed
        cv2.watershed(image_color, markers)
        
        # Use the watershed result as our labeled cells
        labeled_cells = markers.copy()
        labeled_cells[labeled_cells == 1] = 0  # Remove background
        labeled_cells[labeled_cells == -1] = 0  # Remove watershed boundaries
        
        # We'll use labeled_cells for algorithm processing, but preserve original_mask for visualization
        
        # Count number of cells
        num_cells = np.max(labeled_cells)
        
        if num_cells < 2:
            return [], labeled_cells
        
        # Extract properties for each region
        cell_props = regionprops(labeled_cells)
        
        # Filter cells by size
        valid_cells = [prop for prop in cell_props if prop.area >= self.min_cell_size]
        
        if len(valid_cells) < 2:
            return [], labeled_cells
        
        # Store cell info with extended features
        cells = []
        for prop in valid_cells:
            # Get cell center
            center_y, center_x = prop.centroid
            
            # Extract advanced features
            features = self._extract_cell_features(image, labeled_cells, prop)
            
            # Create cell data
            cell = {
                'label': prop.label,
                'center': (center_x, center_y),
                'area': features['area'],
                'perimeter': features['perimeter'],
                'roundness': features['roundness'],
                'eccentricity': features['eccentricity'],
                'intensity': features['intensity_mean'],
                'intensity_std': features['intensity_std'],
                'wall_thickness': features['wall_thickness'],
                'aspect_ratio': features['aspect_ratio'],
                'texture_contrast': features['contrast'],
                'texture_homogeneity': features['homogeneity'],
                'texture_energy': features['energy'],
                'texture_correlation': features['correlation'],
                'bounding_box': prop.bbox
            }
            cells.append(cell)
        
        # Calculate distance matrix between all cells
        centers = np.array([cell['center'] for cell in cells])
        if len(centers) > 1:
            distances = squareform(pdist(centers, 'euclidean'))
        else:
            return [], labeled_cells
        
        # Find potential division events
        division_events = []
        
        for i in range(len(cells)):
            for j in range(i+1, len(cells)):
                # Get distance between cell centers
                distance = distances[i, j]
                
                # Check if cells are close enough to be a division event
                if distance <= self.distance_threshold * 1.5:  # Use a slightly larger threshold for ML
                    # Calculate touching area
                    touch_area = self.calculate_touching_area(labeled_cells, cells[i]['label'], cells[j]['label'])
                    
                    # Identify mother and daughter cells based on size
                    if cells[i]['area'] >= cells[j]['area']:
                        mother_idx, daughter_idx = i, j
                    else:
                        mother_idx, daughter_idx = j, i
                    
                    # Calculate size ratio (daughter to mother)
                    size_ratio = cells[daughter_idx]['area'] / cells[mother_idx]['area']
                    
                    # Extract features for ML-based confidence
                    # Combination of distance, size ratio, shape differences, and texture differences
                    # to determine if this is likely a real division event
                    
                    # Calculate feature differences that are indicative of mother-daughter relationship
                    roundness_diff = abs(cells[mother_idx]['roundness'] - cells[daughter_idx]['roundness'])
                    intensity_diff = abs(cells[mother_idx]['intensity'] - cells[daughter_idx]['intensity'])
                    texture_diff = abs(cells[mother_idx]['texture_contrast'] - cells[daughter_idx]['texture_contrast'])
                    eccentricity_diff = abs(cells[mother_idx]['eccentricity'] - cells[daughter_idx]['eccentricity'])
                    
                    # Calculate advanced confidence score
                    confidence = self._calculate_ml_confidence(
                        distance=distance,
                        size_ratio=size_ratio,
                        touch_area=touch_area,
                        roundness_diff=roundness_diff,
                        intensity_diff=intensity_diff,
                        texture_diff=texture_diff,
                        eccentricity_diff=eccentricity_diff,
                        mother_cell=cells[mother_idx],
                        daughter_cell=cells[daughter_idx]
                    )
                    
                    # Only add events that meet the confidence threshold
                    if confidence >= confidence_threshold:
                        # Create event data
                        event = {
                            'mother_cell': cells[mother_idx],
                            'daughter_cell': cells[daughter_idx],
                            'distance': distance,
                            'size_ratio': size_ratio,
                            'touch_area': touch_area,
                            'confidence': confidence
                        }
                        division_events.append(event)
        
        return division_events, labeled_cells
        
    def _calculate_ml_confidence(self, distance, size_ratio, touch_area, roundness_diff, 
                               intensity_diff, texture_diff, eccentricity_diff,
                               mother_cell, daughter_cell):
        """
        Calculate an advanced confidence score for a division event based on multiple features.
        
        Returns:
        --------
        confidence : float
            Confidence score between 0 and 1
        """
        # Distance factor: even more permissive to catch more potential divisions
        distance_factor = max(0, 1.2 - distance / (self.distance_threshold * 4.0))
        
        # Size ratio factor: much more permissive for yeast buds which can be very small
        ideal_ratio = 0.2  # Allow even smaller daughter cells
        # More permissive on size ratio
        size_ratio_factor = 1.0 if size_ratio <= 0.8 else max(0, 2.0 - size_ratio * 1.5)
        
        # Touch area factor: daughter cells often remain in contact with mother
        # Higher weight on touch area
        smaller_perimeter = min(mother_cell['perimeter'], daughter_cell['perimeter'])
        touch_factor = min(1.0, touch_area / (smaller_perimeter * 0.05)) if smaller_perimeter > 0 else 0.8
        
        # Shape difference factor: budding yeast have distinctive shape differences
        shape_factor = 0.8  # Higher base confidence for shape
        
        # Intensity factor: daughter cells often have different intensity
        intensity_factor = 0.6 + min(0.4, intensity_diff / 20.0)
        
        # Texture factor: more weight on texture differences
        texture_factor = 0.6 + min(0.4, texture_diff * 8.0)
        
        # Eccentricity factor: mother cells tend to be more round than buds
        eccentricity_factor = 0.6 + min(0.4, eccentricity_diff * 3.0)
        
        # Combine factors with weights adjusted to be much more permissive
        confidence = (
            0.20 * distance_factor +     # Distance still important but less weight
            0.25 * size_ratio_factor +   # Size ratio is important for budding
            0.20 * touch_factor +        # More emphasis on touch area for budding yeast
            0.10 * shape_factor +        # Shape differences
            0.10 * intensity_factor +    # Intensity differences
            0.10 * texture_factor +      # Texture differences
            0.05 * eccentricity_factor   # Minor weight on eccentricity
        )
        
        # Boost confidence for clearly adjacent cells
        if distance < (self.distance_threshold * 0.7) and touch_area > 0:
            confidence = min(1.0, confidence * 1.5)  # Stronger boost
        
        # Additional boost for very small cells likely to be buds
        if size_ratio < 0.3 and distance < self.distance_threshold:
            confidence = min(1.0, confidence * 1.3)  # Boost for clear mother-daughter size difference
        
        return min(1.0, max(0.0, confidence))
