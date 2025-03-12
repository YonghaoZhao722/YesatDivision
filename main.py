import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import os
import tifffile
from cell_analysis import CellDivisionAnalyzer
from utils import preprocess_image, create_visualization, format_results, auto_contrast

# Set page configuration
st.set_page_config(
    page_title="Yeast Cell Division Analyzer",
    page_icon="🔬",
    layout="wide"
)

# Session state initialization for persistent analysis results
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'division_events' not in st.session_state:
    st.session_state.division_events = []
if 'labeled_cells' not in st.session_state:
    st.session_state.labeled_cells = None
if 'visualization' not in st.session_state:
    st.session_state.visualization = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'auto_contrast_image' not in st.session_state:
    st.session_state.auto_contrast_image = None

# Initialize parameters to track changes
if 'last_preprocessing_method' not in st.session_state:
    st.session_state.last_preprocessing_method = None
if 'last_distance_threshold' not in st.session_state:
    st.session_state.last_distance_threshold = None
if 'last_size_ratio_threshold' not in st.session_state:
    st.session_state.last_size_ratio_threshold = None
if 'last_min_cell_size' not in st.session_state:
    st.session_state.last_min_cell_size = None
if 'last_cell_detection_method' not in st.session_state:
    st.session_state.last_cell_detection_method = None
if 'last_confidence_threshold' not in st.session_state:
    st.session_state.last_confidence_threshold = None

# App title and description
st.title("Yeast Cell Division Analyzer")
st.markdown("""
This application analyzes yeast cell division events in microscopy images using image processing techniques.
Upload a phase contrast image and its corresponding segmentation mask to detect cell division events.
""")

# Sidebar for parameters
st.sidebar.header("Parameters")
distance_threshold = st.sidebar.slider(
    "Distance Threshold (pixels)",
    min_value=1,
    max_value=150,
    value=75,  # Updated as requested
    help="Maximum distance between cells to be considered as potential division"
)

size_ratio_threshold = st.sidebar.slider(
    "Size Ratio Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.5,  # Updated as requested
    step=0.05,
    help="Mother cells are typically larger than daughter cells (mother/daughter size ratio)"
)

min_cell_size = st.sidebar.slider(
    "Minimum Cell Size (pixels)",
    min_value=10,
    max_value=1000,
    value=50,  # Cell minimum diameter instead of size
    help="Minimum size of a cell region to be considered"
)

st.sidebar.markdown("---")
st.sidebar.header("Advanced Options")
preprocessing_method = st.sidebar.selectbox(
    "Preprocessing Method",
    options=["Basic", "Contrast Enhancement", "Noise Reduction"],
    index=0
)

cell_detection_method = st.sidebar.selectbox(
    "Cell Division Detection Method",
    options=["Distance-Based", "Feature-Based (ML)"],
    index=1,  # Set ML method as default
    help="Distance-Based: Uses physical proximity and size. Feature-Based: Uses machine learning with multiple cell features."
)

# Show confidence threshold only for ML method
if cell_detection_method == "Feature-Based (ML)":
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.20,  # Updated as requested
        step=0.05,
        help="Minimum confidence score to consider a cell division event valid"
    )
else:
    confidence_threshold = 0.20  # Updated as requested

# Main content area - file upload
col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Image")
    original_image = st.file_uploader("Upload phase contrast image", type=["jpg", "jpeg", "png", "tif", "tiff"])
    
    if original_image is not None:
        # Check file extension
        file_ext = original_image.name.split('.')[-1].lower()
        
        # Specialized handling for TIF/TIFF files
        if file_ext in ['tif', 'tiff']:
            # Use tifffile for 16-bit TIFF images
            original_image.seek(0)
            image_array = tifffile.imread(original_image)
            
            # Apply auto-contrast for better visualization (Fiji-like)
            auto_contrast_img = auto_contrast(image_array, clip_percent=0.5)
            
            # Ensure the image is in a displayable format
            if len(auto_contrast_img.shape) == 2:  # grayscale
                st.image(auto_contrast_img, caption="Phase Contrast Image (Auto-contrast)", use_container_width=True)
            else:  # RGB or other
                st.image(auto_contrast_img, caption="Phase Contrast Image (Auto-contrast)", use_container_width=True)
                
            # Ensure grayscale for processing
            if len(image_array.shape) == 3 and image_array.shape[2] > 1:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            # Standard handling for other image formats
            image = Image.open(original_image)
            # Convert to numpy array for processing
            image_array = np.array(image)
            
            # Handle grayscale vs color images
            if len(image_array.shape) == 3 and image_array.shape[2] > 1:
                # Save color image for processing if needed
                color_image = image_array.copy()
                # Convert to grayscale for analysis
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            
            # Apply auto-contrast for better visualization
            auto_contrast_img = auto_contrast(image_array, clip_percent=0.5)
            st.image(auto_contrast_img, caption="Phase Contrast Image (Auto-contrast)", use_container_width=True)

with col2:
    st.subheader("Segmentation Mask")
    mask_image = st.file_uploader("Upload segmentation mask", type=["jpg", "jpeg", "png", "tif", "tiff"])
    
    if mask_image is not None:
        # Check file extension
        file_ext = mask_image.name.split('.')[-1].lower()
        
        # Specialized handling for TIF/TIFF files
        if file_ext in ['tif', 'tiff']:
            # Use tifffile for 16-bit TIFF images
            mask_image.seek(0)
            mask_array = tifffile.imread(mask_image)
            
            # Apply auto contrast to mask for better visualization
            mask_auto_contrast = auto_contrast(mask_array, clip_percent=0.5)
            
            # Apply a colormap (Fire or Rainbow) to the mask for better visualization
            # Create a colored version of the mask using matplotlib's 'hot' colormap (similar to Fiji's 'Fire')
            cmap = plt.cm.get_cmap('hot')  # Similar to Fiji's 'Fire' LUT
            colored_mask = cmap(mask_auto_contrast)
            
            # Convert to uint8 for display
            colored_mask = (colored_mask[:, :, :3] * 255).astype(np.uint8)
            
            # Display the colored mask
            st.image(colored_mask, caption="Segmentation Mask (Fire LUT)", use_container_width=True)
        else:
            # Standard handling for other image formats
            mask = Image.open(mask_image)
            # Convert to numpy array for processing
            mask_array_temp = np.array(mask)
            
            # Convert to grayscale if needed
            if len(mask_array_temp.shape) == 3 and mask_array_temp.shape[2] > 1:
                mask_array_temp = cv2.cvtColor(mask_array_temp, cv2.COLOR_RGB2GRAY)
                
            # Apply auto contrast to mask for better visualization
            mask_auto_contrast = auto_contrast(mask_array_temp, clip_percent=0.5)
            
            # Apply a colormap (Fire or Rainbow) to the mask for better visualization
            # Create a colored version of the mask using matplotlib's 'hot' colormap (similar to Fiji's 'Fire')
            cmap = plt.cm.get_cmap('hot')  # Similar to Fiji's 'Fire' LUT
            colored_mask = cmap(mask_auto_contrast)
            
            # Convert to uint8 for display
            colored_mask = (colored_mask[:, :, :3] * 255).astype(np.uint8)
            
            # Display the colored mask
            st.image(colored_mask, caption="Segmentation Mask (Fire LUT)", use_container_width=True)
            
            # Convert to numpy array for processing
            mask_array = np.array(mask)
        
        # Ensure mask is binary and has the correct data type
        if len(mask_array.shape) == 3 and mask_array.shape[2] > 1:
            mask_array = cv2.cvtColor(mask_array, cv2.COLOR_RGB2GRAY)
            
        # Convert to uint8 to ensure compatibility with OpenCV functions
        if mask_array.dtype != np.uint8:
            if mask_array.dtype == np.uint16:
                # Scale 16-bit to 8-bit
                mask_array = (mask_array / 256).astype(np.uint8)
            elif mask_array.dtype == np.float32 or mask_array.dtype == np.float64:
                # Scale float to 8-bit
                mask_array = (mask_array * 255).astype(np.uint8)
            else:
                # For other types, just convert
                mask_array = mask_array.astype(np.uint8)
                
        # Make sure the mask is binary (0 or 255)
        _, mask_array = cv2.threshold(mask_array, 1, 255, cv2.THRESH_BINARY)
        
        # Optional: Apply morphological operations to ensure clean mask
        kernel = np.ones((3,3), np.uint8)
        mask_array = cv2.morphologyEx(mask_array, cv2.MORPH_CLOSE, kernel)

# Analyze button
analyze_button = st.button("Analyze Cell Division")

# Process image for analysis in the background when both images are uploaded
if original_image is not None and mask_image is not None:
    # Silently process images for analysis
    if 'processed_image' not in st.session_state or preprocessing_method != st.session_state.last_preprocessing_method:
        # Generate preprocessing using the selected method
        processed_image = preprocess_image(image_array, method=preprocessing_method)
        
        # Apply auto-contrast to the original image for better visualization
        auto_contrast_image = auto_contrast(image_array, clip_percent=0.5)
        
        # Store in session state
        st.session_state.processed_image = processed_image
        st.session_state.auto_contrast_image = auto_contrast_image
        st.session_state.last_preprocessing_method = preprocessing_method
    else:
        # Use cached results
        processed_image = st.session_state.processed_image
        auto_contrast_image = st.session_state.auto_contrast_image

# Process images when both are uploaded and button is clicked
if original_image is not None and mask_image is not None and (analyze_button or st.session_state.analyzed):
    # If analyzing for the first time or parameters changed, rerun the analysis
    if (analyze_button or 
        not st.session_state.analyzed or 
        st.session_state.last_distance_threshold != distance_threshold or
        st.session_state.last_size_ratio_threshold != size_ratio_threshold or
        st.session_state.last_min_cell_size != min_cell_size or
        st.session_state.last_preprocessing_method != preprocessing_method or
        st.session_state.last_cell_detection_method != cell_detection_method or
        st.session_state.last_confidence_threshold != confidence_threshold):
        
        with st.spinner("Analyzing cell division events..."):
            try:
                # Use existing processed image if available, otherwise generate it
                if 'processed_image' in st.session_state:
                    processed_image = st.session_state.processed_image
                else:
                    processed_image = preprocess_image(image_array, method=preprocessing_method)
                
                # Initialize analyzer with appropriate method
                analyzer = CellDivisionAnalyzer(
                    distance_threshold=distance_threshold,
                    size_ratio_threshold=size_ratio_threshold,
                    min_cell_size=min_cell_size
                )
                
                # Run analysis based on selected method
                if cell_detection_method == "Distance-Based":
                    division_events, labeled_cells = analyzer.analyze(processed_image, mask_array)
                else:  # "Feature-Based (ML)"
                    # Use the machine learning method with more features
                    division_events, labeled_cells = analyzer.analyze_with_ml(
                        processed_image, 
                        mask_array, 
                        confidence_threshold=confidence_threshold
                    )
                
                # Save to session state for persistence
                st.session_state.analyzed = True
                st.session_state.division_events = division_events
                st.session_state.labeled_cells = labeled_cells
                
                # Create visualization with auto-contrast original image for better visibility
                auto_contrast_image = auto_contrast(image_array, clip_percent=0.5)
                visualization = create_visualization(
                    original_image=auto_contrast_image,  # Use auto-contrast for visualization
                    mask=mask_array, 
                    division_events=division_events,
                    labeled_cells=labeled_cells
                )
                st.session_state.visualization = visualization
                
                # Save current parameters to detect changes
                st.session_state.last_distance_threshold = distance_threshold
                st.session_state.last_size_ratio_threshold = size_ratio_threshold
                st.session_state.last_min_cell_size = min_cell_size
                st.session_state.last_preprocessing_method = preprocessing_method
                st.session_state.last_cell_detection_method = cell_detection_method
                st.session_state.last_confidence_threshold = confidence_threshold
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
    
    # Display results from session state
    st.subheader("Results")
    
    if len(st.session_state.division_events) > 0:
        st.success(f"Found {len(st.session_state.division_events)} potential cell division events")
        
        # Display visualization from session state
        st.image(st.session_state.visualization, caption="Cell Division Events", use_container_width=True)
        
        # Display detailed results
        st.subheader("Detailed Analysis")
        formatted_results = format_results(st.session_state.division_events, st.session_state.labeled_cells)
        st.table(formatted_results)
        
        # Download option for the visualization
        buf = io.BytesIO()
        plt.imsave(buf, st.session_state.visualization)
        buf.seek(0)
        
        st.download_button(
            label="Download Visualization",
            data=buf,
            file_name="cell_division_analysis.png",
            mime="image/png"
        )
    else:
        st.info("No cell division events detected with current parameters. Try adjusting the threshold values.")

# Instructions and information section
with st.expander("How to use this application"):
    st.markdown("""
    ### Instructions:
    
    1. Upload a phase contrast image of yeast cells
    2. Upload the corresponding segmentation mask (binary image with cell regions in white)
    3. Adjust the parameters if necessary:
       - Distance Threshold: Maximum distance between cells to be considered a division event
       - Size Ratio Threshold: Ratio used to differentiate mother and daughter cells
       - Minimum Cell Size: Filters out small artifacts in the image
    4. Click "Analyze Cell Division" to process the images
    5. Review the results and download the visualization if needed
    
    ### Methodology:
    
    This application offers two detection methods:
    
    #### Distance-Based Method:
    - Cell division events are identified when two cells are within the specified distance threshold
    - Mother and daughter cells are differentiated based on size (mothers are typically larger)
    - Basic confidence score calculated from distance and size ratio
    
    #### Feature-Based (ML) Method:
    - Uses multiple cell features to identify division events (recommended for better accuracy)
    - Analyzes texture patterns, cell wall properties, and shape characteristics
    - Calculates confidence score using weighted combination of features
    - Additional features analyzed:
      * Cell wall thickness
      * Texture differences (contrast, homogeneity, energy)
      * Shape characteristics (roundness, eccentricity)
      * Intensity patterns
      * Contact area between cells
    
    ### Tips for better results:
    
    - Ensure the segmentation mask accurately represents cell boundaries
    - Try both detection methods and compare results
    - For Distance-Based method: adjust the distance and size ratio thresholds
    - For ML-Based method: adjust the confidence threshold
    - Use the preprocessing options to improve image quality before analysis
    """)

# Footer
st.markdown("---")
st.markdown("Yeast Cell Division Analyzer | Developed with Streamlit, OpenCV, and scikit-image")
