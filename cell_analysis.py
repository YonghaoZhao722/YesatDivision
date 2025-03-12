import numpy as np
import cv2
from scipy.ndimage import label, center_of_mass, distance_transform_edt
from skimage.measure import regionprops
import pandas as pd

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
