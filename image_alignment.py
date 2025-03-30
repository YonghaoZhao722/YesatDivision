import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import os
import tifffile
import base64
from utils import auto_contrast, apply_fire_lut_to_binary

def app():
    st.title("Image Alignment Tool")
    
    st.markdown("""
    This tool allows you to align DIC/phase contrast images with fluorescence images.
    Upload the images, then use the sliders to shift the DIC/mask image to align with the fluorescence image.
    """)
    
    # Create two columns for file upload
    col1, col2 = st.columns(2)
    
    with col1:
        # File upload section with automatic detection
        st.subheader("DIC/Mask File")
        dic_image = st.file_uploader("Upload DIC or mask image (automatic detection)", type=["jpg", "jpeg", "png", "tif", "tiff"], key="dic_image")
        
        dic_array = None
        mask_array = None
        is_mask = False
        
        if dic_image is not None:
            # Check file extension
            file_ext = dic_image.name.split('.')[-1].lower()
            
            # Specialized handling for TIF/TIFF files
            if file_ext in ['tif', 'tiff']:
                # Use tifffile for 16-bit TIFF images
                dic_image.seek(0)
                dic_array = tifffile.imread(dic_image)
                
                # Check if it looks like a binary/mask image (few unique values)
                unique_values = np.unique(dic_array)
                if len(unique_values) < 10:
                    st.info("Detected binary mask image based on pixel values!")
                    mask_array = dic_array
                    is_mask = True
                    
                    # Apply the 3-3-2 RGB "fire" colormap for visualization
                    if mask_array.dtype == np.uint32 or mask_array.dtype == np.int32:
                        # For uint32 or int32 masks, just convert values > 0 to binary
                        mask_binary = (mask_array > 0).astype(np.uint8) * 255
                        st.info(f"Converted mask from {mask_array.dtype} to binary")
                    elif mask_array.dtype == np.float32 or mask_array.dtype == np.float64:
                        # For float masks, threshold at a small value
                        mask_binary = (mask_array > 0.1).astype(np.uint8) * 255
                        st.info(f"Converted mask from {mask_array.dtype} to binary")
                    else:
                        # For uint8, uint16, etc.
                        mask_threshold = 128 if mask_array.max() > 1 else 0.5
                        mask_binary = (mask_array > mask_threshold).astype(np.uint8) * 255
                    
                    # Use our simplified 3-3-2 RGB Fire LUT for visualization
                    binary_mask = (mask_binary > 0).astype(np.uint8)
                    colored_mask_rgb = apply_fire_lut_to_binary(binary_mask)
                    
                    # Display the colored mask with black background
                    st.image(colored_mask_rgb, caption="Mask (Fire LUT with 3-3-2 RGB mapping)", use_container_width=True)
                else:
                    # This is likely a DIC/phase contrast image
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
                
                # Automatic detection based on histogram
                # Check if it looks like a binary mask (few distinct values)
                if len(np.unique(dic_array)) < 10:
                    st.info("Detected binary mask image based on pixel values!")
                    mask_array = dic_array
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
                    st.image(colored_mask_rgb, caption="Mask (Fire LUT with 3-3-2 RGB mapping)", use_container_width=True)
                else:
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
        # Get image dimensions (using DIC image as reference)
        height, width = dic_array.shape[:2]
        
        # Initialize session state for fine adjustment
        if 'x_shift' not in st.session_state:
            st.session_state.x_shift = 0
        if 'y_shift' not in st.session_state:
            st.session_state.y_shift = 0
        if 'overlay_opacity' not in st.session_state:
            st.session_state.overlay_opacity = 30
        
        # Set up the main layout with controls on left, image on right
        left_col, right_col = st.columns([2, 5])
        
        with left_col:
            st.subheader("Alignment Controls")
            
            # Create sliders for shifting with a smaller range for more precise control
            x_shift = st.slider("Shift X (horizontal)", 
                            float(-min(width//4, 100)), 
                            float(min(width//4, 100)), 
                            float(st.session_state.x_shift),
                            step=0.1)
            
            y_shift = st.slider("Shift Y (vertical)", 
                            float(-min(height//4, 100)), 
                            float(min(height//4, 100)), 
                            float(st.session_state.y_shift),
                            step=0.1)
            
            # Update session state
            st.session_state.x_shift = x_shift
            st.session_state.y_shift = y_shift
            
            # Display current shift values
            st.markdown(f"**Current Shift Values:**")
            st.markdown(f"**X = {st.session_state.x_shift:.1f}, Y = {st.session_state.y_shift:.1f} pixels**")
            
            # Fine adjustment buttons
            st.markdown("### Fine Adjustment")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                if st.button("⬅️", help="Move Left (0.1 pixel)"):
                    st.session_state.x_shift -= 0.1
                    st.rerun()
            
            with col2:
                if st.button("⬆️", help="Move Up (0.1 pixel)"):
                    st.session_state.y_shift -= 0.1
                    st.rerun()
                st.write("")
                st.write("")
                if st.button("⬇️", help="Move Down (0.1 pixel)"):
                    st.session_state.y_shift += 0.1
                    st.rerun()
            
            with col3:
                if st.button("➡️", help="Move Right (0.1 pixel)"):
                    st.session_state.x_shift += 0.1
                    st.rerun()
            
            if st.button("Reset Alignment", help="Reset alignment to center (0,0)"):
                st.session_state.x_shift = 0
                st.session_state.y_shift = 0
                st.rerun()
            
            # Overlay opacity control with a narrower default range
            overlay_opacity = st.slider(
                "Overlay Opacity (%)", 
                min_value=10, 
                max_value=70, 
                value=st.session_state.overlay_opacity,
                help="Control the opacity of the DIC/mask overlay"
            )
            st.session_state.overlay_opacity = overlay_opacity
                
            # Add download button for the overlay image
            if st.button("Generate Overlay for Download"):
                # Prepare the overlay image for download
                try:
                    # Create the overlay by applying the shift and blending
                    # This section is only executed when user wants to download
                    
                    # Prepare fluorescence image
                    if len(fluo_array.shape) == 2:
                        fluo_contrast = auto_contrast(fluo_array, clip_percent=0.5)
                        cmap = plt.cm.get_cmap('hot')
                        fluo_colored = cmap(fluo_contrast)
                        fluo_rgb = (fluo_colored[:, :, :3] * 255).astype(np.uint8)
                    else:
                        fluo_rgb = (auto_contrast(fluo_array) * 255).astype(np.uint8)
                        if len(fluo_rgb.shape) == 3 and fluo_rgb.shape[2] > 3:
                            fluo_rgb = fluo_rgb[:, :, :3]
                    
                    # Prepare DIC/mask image
                    if is_mask:
                        # Create colored mask
                        if len(mask_array.shape) == 2:
                            mask_bin = (mask_array > 0).astype(np.uint8)
                            overlay_rgb = apply_fire_lut_to_binary(mask_bin)
                        else:
                            overlay_rgb = mask_array
                    else:
                        # Use DIC image
                        if len(dic_array.shape) == 2:
                            overlay_rgb = cv2.cvtColor((auto_contrast(dic_array) * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
                        else:
                            overlay_rgb = (auto_contrast(dic_array) * 255).astype(np.uint8)
                    
                    # Apply shift using affine transformation
                    M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
                    shifted_overlay = cv2.warpAffine(overlay_rgb, M, (width, height))
                    
                    # Blend images
                    alpha = overlay_opacity / 100.0
                    beta = 1.0 - alpha
                    overlay = cv2.addWeighted(shifted_overlay, alpha, fluo_rgb, beta, 0)
                    
                    # Add download button
                    buf = io.BytesIO()
                    Image.fromarray(overlay).save(buf, format="PNG")
                    st.download_button(
                        label="Download Overlay Image",
                        data=buf.getvalue(),
                        file_name="aligned_overlay.png",
                        mime="image/png"
                    )
                except Exception as e:
                    st.error(f"Error generating overlay: {e}")
                    st.info("Please try different images or settings.")
        
        with right_col:
            st.subheader("Aligned Overlay")
            
            # Create a more efficient overlay using pure CSS blending and positioning
            
            # First, display both images individually using Streamlit's native image display
            # But we'll make them invisible initially and use HTML/CSS for the actual overlay
            fluo_img_container = st.empty()
            overlay_img_container = st.empty()
            
            # Prepare the base (fluorescence) image
            if len(fluo_array.shape) == 2:
                fluo_contrast = auto_contrast(fluo_array, clip_percent=0.5)
                cmap = plt.cm.get_cmap('hot')
                fluo_colored = cmap(fluo_contrast)
                fluo_rgb = (fluo_colored[:, :, :3] * 255).astype(np.uint8)
            else:
                fluo_rgb = (auto_contrast(fluo_array) * 255).astype(np.uint8)
                if len(fluo_rgb.shape) == 3 and fluo_rgb.shape[2] > 3:
                    fluo_rgb = fluo_rgb[:, :, :3]
            
            # Prepare the overlay (DIC/mask) image
            if is_mask:
                # Create colored mask
                if len(mask_array.shape) == 2:
                    mask_bin = (mask_array > 0).astype(np.uint8)
                    overlay_rgb = apply_fire_lut_to_binary(mask_bin)
                else:
                    overlay_rgb = mask_array
            else:
                # Use DIC image
                if len(dic_array.shape) == 2:
                    overlay_rgb = cv2.cvtColor((auto_contrast(dic_array) * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
                else:
                    overlay_rgb = (auto_contrast(dic_array) * 255).astype(np.uint8)
            
            # Convert to PIL images (only so we can get dimensions)
            fluo_pil = Image.fromarray(fluo_rgb)
            overlay_pil = Image.fromarray(overlay_rgb)
            
            # Get image dimensions
            img_width, img_height = fluo_pil.size
            
            # Save small thumbnails of images to buffer (for lightweight loading)
            # Convert to base64 for embedding in HTML
            fluo_buffer = io.BytesIO()
            fluo_pil.save(fluo_buffer, format="PNG")
            fluo_base64 = base64.b64encode(fluo_buffer.getvalue()).decode()
            
            overlay_buffer = io.BytesIO()
            overlay_pil.save(overlay_buffer, format="PNG")
            overlay_base64 = base64.b64encode(overlay_buffer.getvalue()).decode()
            
            # Create CSS for optimized overlay
            css = f"""
            <style>
                .image-container {{
                    position: relative;
                    width: 100%;
                    max-width: 100%;
                    overflow: hidden;
                }}
                
                .image-container img {{
                    display: block;
                    width: 100%;
                }}
                
                .background-image {{
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    z-index: 1;
                }}
                
                .overlay-image {{
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    opacity: {overlay_opacity/100};
                    transform: translate({x_shift}px, {y_shift}px);
                    z-index: 2;
                    mix-blend-mode: normal;
                }}
            </style>
            """
            
            # Create HTML structure with the images
            html = f"""
            {css}
            <div class="image-container" style="height: {img_height}px;">
                <img 
                    src="data:image/png;base64,{fluo_base64}" 
                    class="background-image" 
                    alt="Fluorescence Image"
                >
                <img 
                    src="data:image/png;base64,{overlay_base64}" 
                    class="overlay-image" 
                    alt="DIC/Mask Image"
                >
            </div>
            """
            
            # Display the overlay using HTML component
            st.components.v1.html(html, height=img_height+20, scrolling=False)
    
    # Instructions/guide
    with st.sidebar:
        st.header("Guide")
        st.markdown("""
        ### How to use this tool
        
        1. **Upload images**:
           - Upload a DIC/phase contrast image or a segmentation mask (automatically detected)
           - Upload a fluorescence image to align with
        
        2. **Adjust alignment**:
           - Use the sliders to shift the DIC/mask image to align with the fluorescence image
           - For fine adjustments, use the arrow buttons to move 0.1 pixel at a time
        
        3. **Customize overlay**:
           - Adjust the opacity of the overlay
           - The system will automatically use the correct image type (DIC or mask)
        
        4. **Download result**:
           - When satisfied with the alignment, click "Generate Overlay for Download"
           - Use the download button to save the aligned overlay
        """)
        
        st.markdown("---")
        
        st.markdown("""
        ### Keyboard Shortcuts
        
        For even finer control, you can use keyboard shortcuts after clicking on the sliders:
        
        - **←/→**: Shift X-axis by 0.1 pixel
        - **↑/↓**: Shift Y-axis by 0.1 pixel
        
        *Note: You need to click on a slider first for keyboard shortcuts to work*
        """)

if __name__ == "__main__":
    app()