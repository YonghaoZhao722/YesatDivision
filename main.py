import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import os
import tifffile
from cell_analysis import CellDivisionAnalyzer
from utils import preprocess_image, create_visualization, format_results

# Set page configuration
st.set_page_config(
    page_title="Yeast Cell Division Analyzer",
    page_icon="🔬",
    layout="wide"
)

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
    max_value=50,
    value=15,
    help="Maximum distance between cells to be considered as potential division"
)

size_ratio_threshold = st.sidebar.slider(
    "Size Ratio Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.7,
    step=0.05,
    help="Mother cells are typically larger than daughter cells (mother/daughter size ratio)"
)

min_cell_size = st.sidebar.slider(
    "Minimum Cell Size (pixels)",
    min_value=10,
    max_value=1000,
    value=100,
    help="Minimum size of a cell region to be considered"
)

st.sidebar.markdown("---")
st.sidebar.header("Advanced Options")
preprocessing_method = st.sidebar.selectbox(
    "Preprocessing Method",
    options=["Basic", "Contrast Enhancement", "Noise Reduction"],
    index=0
)

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
            
            # Display with proper normalization for 16-bit images
            if image_array.dtype == np.uint16:
                display_img = (image_array / 65535 * 255).astype(np.uint8)
            else:
                display_img = image_array.astype(np.uint8)
                
            # Ensure the image is in a displayable format
            if len(display_img.shape) == 2:  # grayscale
                st.image(display_img, caption="Original Phase Contrast Image", use_column_width=True)
            else:  # RGB or other
                st.image(display_img, caption="Original Phase Contrast Image", use_column_width=True)
                
            # Ensure grayscale for processing
            if len(image_array.shape) == 3 and image_array.shape[2] > 1:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            # Standard handling for other image formats
            image = Image.open(original_image)
            # Convert to RGB mode for display
            if image.mode in ['RGBA', 'LA', 'P', 'I', 'I;16']:
                display_img = image.convert('RGB')
            else:
                display_img = image
                
            st.image(display_img, caption="Original Phase Contrast Image", use_column_width=True)
            
            # Convert to numpy array for processing
            image_array = np.array(image)
            
            # Handle grayscale vs color images
            if len(image_array.shape) == 3 and image_array.shape[2] > 1:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

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
            
            # Display with proper normalization for 16-bit images
            if mask_array.dtype == np.uint16:
                display_mask = (mask_array / 65535 * 255).astype(np.uint8)
            else:
                display_mask = mask_array.astype(np.uint8)
                
            # Ensure the image is in a displayable format
            if len(display_mask.shape) == 2:  # grayscale
                st.image(display_mask, caption="Segmentation Mask", use_column_width=True)
            else:  # RGB or other
                st.image(display_mask, caption="Segmentation Mask", use_column_width=True)
        else:
            # Standard handling for other image formats
            mask = Image.open(mask_image)
            # Convert to RGB mode for display
            if mask.mode in ['RGBA', 'LA', 'P', 'I', 'I;16']:
                display_mask = mask.convert('RGB')
            else:
                display_mask = mask
                
            st.image(display_mask, caption="Segmentation Mask", use_column_width=True)
            
            # Convert to numpy array for processing
            mask_array = np.array(mask)
        
        # Ensure mask is binary
        if len(mask_array.shape) == 3 and mask_array.shape[2] > 1:
            mask_array = cv2.cvtColor(mask_array, cv2.COLOR_RGB2GRAY)
        _, mask_array = cv2.threshold(mask_array, 127, 255, cv2.THRESH_BINARY)

# Analyze button
analyze_button = st.button("Analyze Cell Division")

# Process images when both are uploaded and button is clicked
if original_image is not None and mask_image is not None and analyze_button:
    with st.spinner("Analyzing cell division events..."):
        try:
            # Preprocess images
            processed_image = preprocess_image(image_array, method=preprocessing_method)
            
            # Initialize analyzer
            analyzer = CellDivisionAnalyzer(
                distance_threshold=distance_threshold,
                size_ratio_threshold=size_ratio_threshold,
                min_cell_size=min_cell_size
            )
            
            # Run analysis
            division_events, labeled_cells = analyzer.analyze(processed_image, mask_array)
            
            # Display results
            st.subheader("Results")
            
            if len(division_events) > 0:
                st.success(f"Found {len(division_events)} potential cell division events")
                
                # Create visualization
                visualization = create_visualization(
                    original_image=image_array,
                    mask=mask_array, 
                    division_events=division_events,
                    labeled_cells=labeled_cells
                )
                
                st.image(visualization, caption="Cell Division Events", use_container_width=True)
                
                # Display detailed results
                st.subheader("Detailed Analysis")
                formatted_results = format_results(division_events, labeled_cells)
                st.table(formatted_results)
                
                # Download option for the visualization
                buf = io.BytesIO()
                plt.imsave(buf, visualization)
                buf.seek(0)
                
                st.download_button(
                    label="Download Visualization",
                    data=buf,
                    file_name="cell_division_analysis.png",
                    mime="image/png"
                )
            else:
                st.info("No cell division events detected with current parameters. Try adjusting the threshold values.")
        
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")

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
    
    - Cell division events are identified when two cells are within the specified distance threshold
    - Mother and daughter cells are differentiated based on size (mothers are typically larger)
    - Additional features like contact area and shape may be used for classification
    
    ### Tips for better results:
    
    - Ensure the segmentation mask accurately represents cell boundaries
    - Adjust the distance threshold based on the image resolution and cell density
    - The size ratio threshold can be modified based on the specific yeast strain or growth conditions
    """)

# Footer
st.markdown("---")
st.markdown("Yeast Cell Division Analyzer | Developed with Streamlit, OpenCV, and scikit-image")
