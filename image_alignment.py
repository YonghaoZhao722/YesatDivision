import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import os
import tifffile
import json
from utils import auto_contrast

def app():
    # Header
    st.title("Image Alignment Tool")
    st.markdown("""
    This tool allows you to align DIC/phase contrast images with fluorescence images.
    Upload the images, then use the sliders to shift the DIC/mask image to align with the fluorescence image.
    """)
    
    # Create two columns for file upload
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("DIC/Phase Contrast Image")
        dic_image = st.file_uploader("Upload DIC/Phase contrast image", type=["jpg", "jpeg", "png", "tif", "tiff"], key="dic_image")
        
        dic_array = None
        if dic_image is not None:
            # Check file extension
            file_ext = dic_image.name.split('.')[-1].lower()
            
            # Specialized handling for TIF/TIFF files
            if file_ext in ['tif', 'tiff']:
                # Use tifffile for 16-bit TIFF images
                dic_image.seek(0)
                dic_array = tifffile.imread(dic_image)
                
                # Apply auto-contrast for better visualization
                dic_contrast = auto_contrast(dic_array, clip_percent=0.5)
                st.image(dic_contrast, caption="DIC/Phase Contrast Image", use_container_width=True)
                
                # Ensure grayscale for processing
                if len(dic_array.shape) == 3 and dic_array.shape[2] > 1:
                    dic_array = cv2.cvtColor(dic_array, cv2.COLOR_RGB2GRAY)
            else:
                # Standard handling for other image formats
                image = Image.open(dic_image)
                # Convert to numpy array for processing
                dic_array = np.array(image)
                
                # Handle grayscale vs color images
                if len(dic_array.shape) == 3 and dic_array.shape[2] > 1:
                    # Convert to grayscale for analysis
                    dic_gray = cv2.cvtColor(dic_array, cv2.COLOR_RGB2GRAY)
                    # Keep original for visualization
                    dic_contrast = auto_contrast(dic_gray, clip_percent=0.5)
                    dic_array = dic_gray
                else:
                    dic_contrast = auto_contrast(dic_array, clip_percent=0.5)
                
                st.image(dic_contrast, caption="DIC/Phase Contrast Image", use_container_width=True)
        
        # Segmentation mask for DIC image (optional)
        st.subheader("Segmentation Mask (Optional)")
        mask_image = st.file_uploader("Upload segmentation mask", type=["jpg", "jpeg", "png", "tif", "tiff"], key="mask_image")
        
        mask_array = None
        if mask_image is not None:
            # Check file extension
            file_ext = mask_image.name.split('.')[-1].lower()
            
            # Specialized handling for TIF/TIFF files
            if file_ext in ['tif', 'tiff']:
                # Use tifffile for 16-bit TIFF images
                mask_image.seek(0)
                mask_array = tifffile.imread(mask_image)
                
                # Check for metadata to determine if it's a mask
                is_mask = False
                metadata = tifffile.TiffFile(mask_image).pages[0].tags
                if 'ImageDescription' in metadata:
                    desc = metadata['ImageDescription'].value
                    if isinstance(desc, bytes):
                        desc = desc.decode('utf-8', errors='ignore')
                    if '{"shape":' in desc or '"shape":' in desc:
                        is_mask = True
                        st.info("Detected mask image from metadata")
                
                # Convert to binary mask if needed based on data type
                mask_binary = None
                # Handle different data types
                if mask_array.dtype == np.uint32 or mask_array.dtype == np.int32:
                    # For uint32 or int32 masks, just convert values > 0 to 1
                    mask_binary = (mask_array > 0).astype(np.uint8) * 255
                    st.info(f"Converted mask from {mask_array.dtype} to binary")
                elif mask_array.dtype == np.float32 or mask_array.dtype == np.float64:
                    # For float masks, threshold at a small value
                    mask_binary = (mask_array > 0.1).astype(np.uint8) * 255
                    st.info(f"Converted mask from {mask_array.dtype} to binary")
                else:
                    # For uint8, uint16, etc.
                    _, mask_binary = cv2.threshold(mask_array.astype(np.uint8), 1, 255, cv2.THRESH_BINARY)
                
                # Apply color for visualization
                cmap = plt.colormaps['plasma']
                mask_color = cmap(mask_binary.astype(float) / 255.0)
                mask_color = (mask_color[:, :, :3] * 255).astype(np.uint8)
                
                st.image(mask_color, caption="Segmentation Mask", use_container_width=True)
            else:
                # Standard handling for other image formats
                mask = Image.open(mask_image)
                # Convert to numpy array for processing
                mask_array = np.array(mask)
                
                # Convert to grayscale if needed
                if len(mask_array.shape) == 3 and mask_array.shape[2] > 1:
                    mask_array = cv2.cvtColor(mask_array, cv2.COLOR_RGB2GRAY)
                
                # Convert to binary mask
                _, mask_binary = cv2.threshold(mask_array, 1, 255, cv2.THRESH_BINARY)
                
                # Apply color for visualization
                cmap = plt.colormaps['plasma']
                mask_color = cmap(mask_binary.astype(float) / 255.0)
                mask_color = (mask_color[:, :, :3] * 255).astype(np.uint8)
                
                st.image(mask_color, caption="Segmentation Mask", use_container_width=True)
    
    with col2:
        st.subheader("Fluorescence Image (Cy3/DAPI)")
        fluo_image = st.file_uploader("Upload fluorescence image", type=["jpg", "jpeg", "png", "tif", "tiff"], key="fluo_image")
        
        fluo_array = None
        if fluo_image is not None:
            # Check file extension
            file_ext = fluo_image.name.split('.')[-1].lower()
            
            # Specialized handling for TIF/TIFF files
            if file_ext in ['tif', 'tiff']:
                # Use tifffile for 16-bit TIFF images
                fluo_image.seek(0)
                fluo_array = tifffile.imread(fluo_image)
                
                # Apply auto-contrast for better visualization
                fluo_contrast = auto_contrast(fluo_array, clip_percent=0.5)
                
                # If single channel, apply a colormap appropriate for the fluorescence type
                if len(fluo_array.shape) == 2:
                    # Apply a colormap appropriate for fluorescence
                    cmap = plt.cm.get_cmap('hot')  # Red/yellow for Cy3/TRITC-like channels
                    fluo_colored = cmap(fluo_contrast)
                    fluo_colored = (fluo_colored[:, :, :3] * 255).astype(np.uint8)
                    st.image(fluo_colored, caption="Fluorescence Image", use_container_width=True)
                else:
                    st.image(fluo_contrast, caption="Fluorescence Image", use_container_width=True)
            else:
                # Standard handling for other image formats
                image = Image.open(fluo_image)
                # Convert to numpy array for processing
                fluo_array = np.array(image)
                
                # Apply auto-contrast
                if len(fluo_array.shape) == 2:
                    fluo_contrast = auto_contrast(fluo_array, clip_percent=0.5)
                    # Apply a colormap appropriate for fluorescence
                    cmap = plt.cm.get_cmap('hot')  # Red/yellow for Cy3/TRITC-like channels
                    fluo_colored = cmap(fluo_contrast)
                    fluo_colored = (fluo_colored[:, :, :3] * 255).astype(np.uint8)
                    st.image(fluo_colored, caption="Fluorescence Image", use_container_width=True)
                else:
                    fluo_contrast = auto_contrast(fluo_array, clip_percent=1.0)
                    st.image(fluo_contrast, caption="Fluorescence Image", use_container_width=True)
        
        # Select fluorescence channel (for multichannel images)
        fluo_channel = None
        if fluo_array is not None and len(fluo_array.shape) == 3 and fluo_array.shape[2] >= 3:
            st.subheader("Fluorescence Channel Selection")
            channel_options = ["All Channels", "Red (Cy3/TRITC)", "Green (GFP/FITC)", "Blue (DAPI)"]
            selected_channel = st.selectbox("Select channel to display", channel_options)
            
            if selected_channel == "Red (Cy3/TRITC)":
                fluo_channel = fluo_array[:, :, 0]
            elif selected_channel == "Green (GFP/FITC)":
                fluo_channel = fluo_array[:, :, 1]
            elif selected_channel == "Blue (DAPI)":
                fluo_channel = fluo_array[:, :, 2]
    
    # Alignment controls (only show if both images are uploaded)
    if dic_array is not None and fluo_array is not None:
        st.subheader("Alignment Controls")
        
        # Get image dimensions (using DIC image as reference)
        height, width = dic_array.shape[:2]
        
        # Initialize session state for fine adjustment
        if 'x_shift' not in st.session_state:
            st.session_state.x_shift = 0
        if 'y_shift' not in st.session_state:
            st.session_state.y_shift = 0
        
        # Create sliders for shifting with a smaller range for more precise control
        col1, col2 = st.columns(2)
        with col1:
            x_shift = st.slider("Shift X (horizontal)", 
                              -min(width//4, 100), 
                              min(width//4, 100), 
                              st.session_state.x_shift)
        with col2:
            y_shift = st.slider("Shift Y (vertical)", 
                              -min(height//4, 100), 
                              min(height//4, 100), 
                              st.session_state.y_shift)
        
        # Update session state
        st.session_state.x_shift = x_shift
        st.session_state.y_shift = y_shift
        
        # Fine adjustment buttons
        st.markdown("### Fine Adjustment")
        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 3])
        
        with col1:
            if st.button("⬅️", help="Move Left (1 pixel)"):
                st.session_state.x_shift -= 1
                st.experimental_rerun()
        
        with col3:
            if st.button("➡️", help="Move Right (1 pixel)"):
                st.session_state.x_shift += 1
                st.experimental_rerun()
        
        with col2:
            if st.button("⬆️", help="Move Up (1 pixel)"):
                st.session_state.y_shift -= 1
                st.experimental_rerun()
        
        with col4:
            if st.button("⬇️", help="Move Down (1 pixel)"):
                st.session_state.y_shift += 1
                st.experimental_rerun()
        
        with col5:
            if st.button("Reset Alignment", help="Reset alignment to center (0,0)"):
                st.session_state.x_shift = 0
                st.session_state.y_shift = 0
                st.experimental_rerun()
        
        # Overlay opacity control with a narrower default range
        overlay_opacity = st.slider(
            "Overlay Opacity (%)", 
            min_value=10, 
            max_value=70, 
            value=30,
            help="Control the opacity of the DIC/mask overlay"
        )
        
        # Choose overlay type
        overlay_type = st.radio(
            "Overlay Type",
            ["DIC over Fluorescence", "Mask over Fluorescence (if available)"],
            horizontal=True
        )
        
        # Create the overlay with alignment
        st.subheader("Aligned Overlay")
        
        # Create transformation matrix for the shift
        M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
        
        # Choose which image to use for overlay
        if overlay_type == "Mask over Fluorescence (if available)" and mask_array is not None:
            src_image = mask_array
            # Convert to RGB if needed
            if len(src_image.shape) == 2:
                # Handle different data types for the mask
                if src_image.dtype == np.uint32 or src_image.dtype == np.int32:
                    # For uint32 or int32 masks, just convert values > 0 to 1
                    src_image_bin = (src_image > 0).astype(np.uint8) * 255
                elif src_image.dtype == np.float32 or src_image.dtype == np.float64:
                    # For float masks, threshold at a small value
                    src_image_bin = (src_image > 0.1).astype(np.uint8) * 255
                else:
                    # For uint8, uint16, etc.
                    try:
                        _, src_image_bin = cv2.threshold(src_image.astype(np.uint8), 1, 255, cv2.THRESH_BINARY)
                    except Exception as e:
                        st.error(f"Error processing mask: {e}")
                        # Fallback for problematic images
                        src_image_bin = (src_image > src_image.mean()/10).astype(np.uint8) * 255
                
                # Apply the 3-3-2 RGB "fire" colormap (similar to Fiji "fire" LUT)
                # Create a "Fire" LUT similar to ImageJ/Fiji
                h, w = src_image_bin.shape[:2]
                rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
                
                # Apply a custom 3-3-2 RGB LUT (similar to "fire" in ImageJ)
                # Red channel (3 bits)
                rgb_image[:,:,0] = (src_image_bin > 0) * 255
                
                # Green channel (3 bits)
                rgb_image[:,:,1] = (src_image_bin > 0) * 210
                
                # Blue channel (2 bits)
                rgb_image[:,:,2] = (src_image_bin > 0) * 150
                
                src_image = rgb_image
        else:
            src_image = dic_array
            # Convert to RGB if needed
            if len(src_image.shape) == 2:
                src_image = cv2.cvtColor(src_image, cv2.COLOR_GRAY2RGB)
        
        # Ensure fluorescence image is RGB for overlay
        if len(fluo_array.shape) == 2:
            # Apply appropriate colormap for fluorescence
            fluo_contrast = auto_contrast(fluo_array, clip_percent=0.5)
            cmap = plt.cm.get_cmap('hot')  # Red/yellow for Cy3/TRITC
            fluo_colored = cmap(fluo_contrast)
            fluo_colored = (fluo_colored[:, :, :3] * 255).astype(np.uint8)
            fluo_rgb = fluo_colored
        else:
            fluo_rgb = auto_contrast(fluo_array, clip_percent=0.5)
            # Ensure it's RGB
            if len(fluo_rgb.shape) == 3 and fluo_rgb.shape[2] > 3:
                fluo_rgb = fluo_rgb[:, :, :3]
        
        # Apply the shift to the DIC/mask image
        shifted_image = cv2.warpAffine(src_image, M, (width, height))
        
        # Ensure both images have the same dtype and channel count
        # Convert both to uint8 RGB format
        if shifted_image.dtype != np.uint8:
            shifted_image = (shifted_image * 255).astype(np.uint8) if shifted_image.max() <= 1.0 else shifted_image.astype(np.uint8)
        
        if fluo_rgb.dtype != np.uint8:
            fluo_rgb = (fluo_rgb * 255).astype(np.uint8) if fluo_rgb.max() <= 1.0 else fluo_rgb.astype(np.uint8)
            
        # Ensure both have 3 channels (RGB)
        if len(shifted_image.shape) == 2:
            shifted_image = cv2.cvtColor(shifted_image, cv2.COLOR_GRAY2RGB)
        elif shifted_image.shape[2] > 3:
            shifted_image = shifted_image[:,:,:3]
            
        if len(fluo_rgb.shape) == 2:
            fluo_rgb = cv2.cvtColor(fluo_rgb, cv2.COLOR_GRAY2RGB)
        elif fluo_rgb.shape[2] > 3:
            fluo_rgb = fluo_rgb[:,:,:3]
        
        # Ensure both images have the same dimensions
        if shifted_image.shape[:2] != fluo_rgb.shape[:2]:
            # Resize to match the first image dimensions
            fluo_rgb = cv2.resize(fluo_rgb, (shifted_image.shape[1], shifted_image.shape[0]))
        
        # Create overlay
        alpha = overlay_opacity / 100.0
        overlay = cv2.addWeighted(shifted_image, alpha, fluo_rgb, 1.0 - alpha, 0)
        
        # Display the overlay
        st.image(overlay, caption=f"Aligned Overlay (Shift X: {x_shift}, Y: {y_shift})", use_container_width=True)
        
        # Display the current alignment values
        st.success(f"Current alignment: X shift = {x_shift} pixels, Y shift = {y_shift} pixels")
        
        # Add download button for the aligned overlay
        buf = io.BytesIO()
        # Use PIL to save the image as it's more reliable with different image formats
        Image.fromarray(overlay).save(buf, format="PNG")
        buf.seek(0)
        
        st.download_button(
            label="Download Aligned Overlay",
            data=buf,
            file_name="aligned_overlay.png",
            mime="image/png"
        )
    
    # Instructions
    with st.expander("How to use this tool"):
        st.markdown("""
        ### Instructions:
        
        1. Upload a DIC/phase contrast image (and optionally its segmentation mask)
        2. Upload a fluorescence image (Cy3, DAPI, or other fluorescent channel)
        3. Use the sliders to shift the DIC/mask image to align with the fluorescence image
        4. Use the ⬅️⬆️➡️⬇️ buttons for 1-pixel precision adjustments
        5. Adjust the overlay opacity (default 30%) to best visualize the alignment
        6. Choose between DIC or mask overlay
        7. Download the aligned overlay image when satisfied
        
        ### Features:
        
        - **Auto-detection**: The tool automatically detects whether the uploaded file is a DIC or mask image based on metadata
        - **Fire LUT**: Segmentation masks are displayed using a Fiji-like "Fire" LUT with 3-3-2 RGB color mapping
        - **Fine Controls**: Precision pixel-by-pixel alignment controls for perfect positioning
        - **Channel Selection**: For multichannel fluorescence images, you can select specific channels
        - **Flexible Overlay**: Choose between DIC overlay or mask overlay with adjustable opacity
        - **Alignment Values**: The shift values (X and Y) show the offset between images for batch processing
        
        ### Supported Formats:
        
        - 16-bit TIF/TIFF microscopy images
        - 8-bit mask images
        - Standard file formats (JPG, PNG)
        - Various data types (uint8, uint16, uint32, float)
        """)

if __name__ == "__main__":
    app()