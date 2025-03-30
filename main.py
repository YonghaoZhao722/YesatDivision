import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import os
import tifffile
from cell_analysis import CellDivisionAnalyzer
from utils import preprocess_image, create_visualization, format_results, auto_contrast, apply_fire_lut_to_binary

def app():
    # Initialize session state for this page
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

    # Add overlay opacity control
    overlay_opacity = st.sidebar.slider(
        "Overlay Opacity (%)",
        min_value=10,
        max_value=100,
        value=30,  # Default 30% as requested
        help="Control the opacity of the cell mask overlay"
    )

    st.sidebar.markdown("---")
    st.sidebar.header("Advanced Options")
    preprocessing_method = st.sidebar.selectbox(
        "Preprocessing Method",
        options=["Basic", "Contrast Enhancement", "Noise Reduction"],
        index=0
    )

    # Add segmentation options
    st.sidebar.markdown("### Segmentation Options")
    apply_watershed = st.sidebar.checkbox(
        "Apply Watershed Segmentation", 
        value=True,
        help="Use watershed algorithm to separate touching cells"
    )

    apply_thresholding = st.sidebar.checkbox(
        "Apply Binary Thresholding", 
        value=True,
        help="Convert mask to binary (0 or 255)"
    )

    apply_morphology = st.sidebar.checkbox(
        "Apply Morphological Operations", 
        value=True,
        help="Apply closing operation to clean up mask"
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

    # Main content area - file upload with automatic detection
    st.subheader("DIC/Mask File")
    dic_image = st.file_uploader("Upload DIC or mask image (automatic detection)", type=["jpg", "jpeg", "png", "tif", "tiff"])
    
    # Variables to store our image arrays
    image_array = None
    mask_array = None
    is_mask = False
    
    if dic_image is not None:
        # Check file extension
        file_ext = dic_image.name.split('.')[-1].lower()
        
        # Specialized handling for TIF/TIFF files
        if file_ext in ['tif', 'tiff']:
            # Use tifffile for 16-bit TIFF images
            dic_image.seek(0)
            img_array = tifffile.imread(dic_image)
            
            # Check for metadata to determine if it's a mask
            is_mask = False
            try:
                metadata = tifffile.TiffFile(dic_image).pages[0].tags
                if 'ImageDescription' in metadata:
                    desc = metadata['ImageDescription'].value
                    if isinstance(desc, bytes):
                        desc = desc.decode('utf-8', errors='ignore')
                    if '{"shape":' in desc or '"shape":' in desc:
                        is_mask = True
                        st.info("Detected mask image from metadata!")
                        mask_array = img_array
            except Exception as e:
                st.warning(f"Could not read metadata: {e}")
            
            # Auto-detection based on image properties if metadata doesn't help
            if not is_mask:
                # Check if it looks like a binary/mask image (few unique values)
                unique_values = np.unique(img_array)
                if len(unique_values) < 10:
                    st.info("Detected binary mask image based on pixel values!")
                    mask_array = img_array
                    is_mask = True
                else:
                    # This is likely a DIC/phase contrast image
                    image_array = img_array
            
            # Process based on detected type
            if is_mask:
                # Handle mask processing
                # Handle different data types for the mask
                if mask_array.dtype == np.uint32 or mask_array.dtype == np.int32:
                    # For uint32 or int32 masks, just convert values > 0 to binary
                    mask_binary = (mask_array > 0).astype(np.uint8) * 255
                    st.info(f"Converted mask from {mask_array.dtype} to binary")
                    # Create binary version for processing
                    mask_array = mask_array > 0
                elif mask_array.dtype == np.float32 or mask_array.dtype == np.float64:
                    # For float masks, threshold at a small value
                    mask_binary = (mask_array > 0.1).astype(np.uint8) * 255
                    st.info(f"Converted mask from {mask_array.dtype} to binary")
                    # Create binary version for processing
                    mask_array = mask_array > 0.1
                else:
                    # For uint8, uint16, etc.
                    mask_threshold = 128 if mask_array.max() > 1 else 0.5
                    mask_binary = (mask_array > mask_threshold).astype(np.uint8) * 255
                    # Create binary version for processing
                    mask_array = mask_array > mask_threshold
                
                # Apply auto contrast to mask for better visualization
                mask_auto_contrast = mask_binary
                
                # Use our simplified 3-3-2 RGB Fire LUT for visualization
                binary_mask = (mask_auto_contrast > 0).astype(np.uint8)
                colored_mask_rgb = apply_fire_lut_to_binary(binary_mask)
                
                # Display the colored mask with black background
                st.image(colored_mask_rgb, caption="Segmentation Mask (Fire LUT with 3-3-2 RGB mapping)", use_container_width=True)
            else:
                # Process as DIC/phase contrast image
                # Apply auto-contrast for better visualization (Fiji-like)
                auto_contrast_img = auto_contrast(image_array, clip_percent=0.5)
                
                # Ensure the image is in a displayable format
                st.image(auto_contrast_img, caption="Phase Contrast Image (Auto-contrast)", use_container_width=True)
                
                # Ensure grayscale for processing
                if len(image_array.shape) == 3 and image_array.shape[2] > 1:
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            # Standard handling for other image formats
            image = Image.open(dic_image)
            # Convert to numpy array for processing
            img_array = np.array(image)
            
            # Automatic detection based on histogram
            # Check if it looks like a binary mask (few distinct values)
            if len(np.unique(img_array)) < 10:
                st.info("Detected binary mask image based on pixel values!")
                mask_array = img_array
                is_mask = True
                
                # Process as mask
                if len(mask_array.shape) == 3 and mask_array.shape[2] > 1:
                    mask_array = cv2.cvtColor(mask_array, cv2.COLOR_RGB2GRAY)
                
                # Apply binary threshold to ensure it's a proper mask
                _, mask_binary = cv2.threshold(mask_array.astype(np.uint8), 1, 255, cv2.THRESH_BINARY)
                
                # Use our simplified 3-3-2 RGB Fire LUT for visualization
                binary_mask = (mask_binary > 0).astype(np.uint8)
                colored_mask_rgb = apply_fire_lut_to_binary(binary_mask)
                
                # Display the colored mask with black background
                st.image(colored_mask_rgb, caption="Segmentation Mask (Fire LUT with 3-3-2 RGB mapping)", use_container_width=True)
            else:
                # Process as DIC/phase contrast image
                image_array = img_array
                is_mask = False
                
                # Handle grayscale vs color images
                if len(image_array.shape) == 3 and image_array.shape[2] > 1:
                    # Save color image for processing if needed
                    color_image = image_array.copy()
                    # Convert to grayscale for analysis
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
                
                # Apply auto-contrast for better visualization
                auto_contrast_img = auto_contrast(image_array, clip_percent=0.5)
                st.image(auto_contrast_img, caption="Phase Contrast Image (Auto-contrast)", use_container_width=True)
    
    # Display upload information
    if dic_image is not None:
        if is_mask:
            st.info("📄 Uploaded image detected as a mask. Please upload a DIC/phase contrast image to analyze cell division.")
        else:
            st.info("📄 Uploaded image detected as DIC/phase contrast. Please upload a mask to analyze cell division.")
    else:
        st.info("⬆️ Please upload a file. The app will automatically detect whether it's a DIC or mask image.")
    
    # Ensure mask is binary and has the correct data type if it exists
    if mask_array is not None:
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

            # Apply binary thresholding if selected
            if apply_thresholding:
                _, mask_array = cv2.threshold(mask_array, 1, 255, cv2.THRESH_BINARY)

            # Apply morphological operations if selected
            if apply_morphology:
                kernel = np.ones((3,3), np.uint8)
                mask_array = cv2.morphologyEx(mask_array, cv2.MORPH_CLOSE, kernel)

    # Analyze button
    analyze_button = st.button("Analyze Cell Division")

    # Process image for analysis in the background when both images are uploaded
    if image_array is not None and mask_array is not None:
        # Silently process images for analysis
        if ('processed_image' not in st.session_state or 
            preprocessing_method != st.session_state.last_preprocessing_method or
            'last_overlay_opacity' not in st.session_state or
            st.session_state.last_overlay_opacity != overlay_opacity):

            # Generate preprocessing using the selected method
            processed_image = preprocess_image(image_array, method=preprocessing_method)

            # Apply auto-contrast to the original image for better visualization
            auto_contrast_image = auto_contrast(image_array, clip_percent=0.5)

            # Create labeled image (without division markers)
            from skimage.measure import label
            binary_mask = mask_array > 0
            labeled_mask = label(binary_mask)

            # Create a Fiji-style overlay of the mask on the original image
            overlay_image = create_visualization(
                original_image=auto_contrast_image,
                mask=mask_array,
                division_events=[],  # No division events here
                labeled_cells=labeled_mask,
                overlay_opacity=overlay_opacity/100.0  # Convert percentage to decimal
            )

            # Store in session state
            st.session_state.processed_image = processed_image
            st.session_state.auto_contrast_image = auto_contrast_image
            st.session_state.overlay_image = overlay_image
            st.session_state.last_preprocessing_method = preprocessing_method
            st.session_state.last_overlay_opacity = overlay_opacity
        else:
            # Use cached results
            processed_image = st.session_state.processed_image
            auto_contrast_image = st.session_state.auto_contrast_image
            overlay_image = st.session_state.overlay_image

        # Show the overlay of mask on original image
        st.subheader("Segmentation Overlay")
        st.image(st.session_state.overlay_image, caption=f"Overlay with {overlay_opacity}% opacity", use_container_width=True)

    # Process images when both are uploaded and button is clicked
    if image_array is not None and mask_array is not None and (analyze_button or st.session_state.analyzed):
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
                            confidence_threshold=confidence_threshold,
                            apply_watershed=apply_watershed
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
                        labeled_cells=labeled_cells,
                        overlay_opacity=overlay_opacity/100.0  # Convert percentage to decimal
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
            # Ensure visualization exists and is correctly formatted for PIL
            if 'visualization' in st.session_state and st.session_state.visualization is not None:
                # Convert to uint8 if needed
                vis_img = st.session_state.visualization
                if vis_img.dtype != np.uint8:
                    if vis_img.max() <= 1.0:
                        vis_img = (vis_img * 255).astype(np.uint8)
                    else:
                        vis_img = np.clip(vis_img, 0, 255).astype(np.uint8)
                # Use PIL to save the image as it's more reliable with different image formats
                Image.fromarray(vis_img).save(buf, format="PNG")
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

if __name__ == "__main__":
    # Set page configuration
    st.set_page_config(
        page_title="Yeast Cell Division Analyzer",
        page_icon="🔬",
        layout="wide"
    )
    app()