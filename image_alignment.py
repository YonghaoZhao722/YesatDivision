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
                            -min(width//4, 100), 
                            min(width//4, 100), 
                            float(st.session_state.x_shift),
                            step=0.1)
            
            y_shift = st.slider("Shift Y (vertical)", 
                            -min(height//4, 100), 
                            min(height//4, 100), 
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
            
            # Default overlay type - automatic based on what was detected
            if is_mask:
                overlay_type = "Mask over Fluorescence (if available)" 
            else:
                overlay_type = "DIC over Fluorescence"
                
            # Add download button for the overlay image
            if st.button("Generate Overlay for Download"):
                # Generate the overlay image only when requested
                try:
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
                            # Handle uint32 which is not supported by cvtColor
                            if src_image.dtype == np.uint32 or src_image.dtype == np.int32:
                                # Convert to uint8 first
                                src_image = (src_image > 0).astype(np.uint8) * 255
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
                        # Handle various data types safely
                        if shifted_image.dtype == np.uint32 or shifted_image.dtype == np.int32:
                            # For uint32/int32, first convert to a format OpenCV can handle
                            shifted_image = (shifted_image > 0).astype(np.uint8) * 255
                        elif shifted_image.max() <= 1.0:
                            shifted_image = (shifted_image * 255).astype(np.uint8)
                        else:
                            # Scale down larger bit depths
                            if shifted_image.dtype == np.uint16:
                                shifted_image = (shifted_image / 256).astype(np.uint8)
                            else:
                                shifted_image = shifted_image.astype(np.uint8)
                    
                    # Ensure shifted image is RGB
                    if len(shifted_image.shape) == 2:
                        shifted_image = cv2.cvtColor(shifted_image, cv2.COLOR_GRAY2RGB)
                    elif shifted_image.shape[2] == 4:  # With alpha channel
                        shifted_image = shifted_image[:, :, :3]
                    
                    # Ensure fluo_rgb is RGB
                    if len(fluo_rgb.shape) == 2:
                        fluo_rgb = cv2.cvtColor(fluo_rgb, cv2.COLOR_GRAY2RGB)
                    elif isinstance(fluo_rgb, np.ndarray) and fluo_rgb.shape[2] == 4:  # With alpha channel
                        fluo_rgb = fluo_rgb[:, :, :3]
                    
                    # Create the blend with controlled opacity
                    alpha = overlay_opacity / 100.0
                    beta = 1.0 - alpha
                    
                    # Ensure both images have the same dimensions
                    if shifted_image.shape[:2] != fluo_rgb.shape[:2]:
                        # Resize to match
                        shifted_image = cv2.resize(shifted_image, (fluo_rgb.shape[1], fluo_rgb.shape[0]))
                    
                    # Convert both images to uint8 RGB if they're not already
                    if shifted_image.dtype != np.uint8:
                        shifted_image = shifted_image.astype(np.uint8)
                    if fluo_rgb.dtype != np.uint8:
                        fluo_rgb = fluo_rgb.astype(np.uint8)
                        
                    # Create the overlay
                    overlay = cv2.addWeighted(shifted_image, alpha, fluo_rgb, beta, 0)
                    
                    # Save to session state for download
                    st.session_state.overlay_image = overlay
                    
                    # Enable download button
                    buf = io.BytesIO()
                    Image.fromarray(overlay).save(buf, format="PNG")
                    st.download_button(
                        label="Download Overlay Image",
                        data=buf.getvalue(),
                        file_name="aligned_overlay.png",
                        mime="image/png"
                    )
                    
                except Exception as e:
                    st.error(f"Error creating overlay: {e}")
                    st.info("Try adjusting the images or checking their formats for compatibility.")
        
        with right_col:
            st.subheader("Aligned Overlay")
            
            # Prepare images for display in the browser
            # For fluorescence image
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
            
            # For DIC/mask image
            if is_mask and mask_array is not None:
                # Process mask image
                src_image = mask_array
                # Convert to RGB if needed
                if len(src_image.shape) == 2:
                    # Convert to binary
                    if src_image.dtype == np.uint32 or src_image.dtype == np.int32:
                        src_image_bin = (src_image > 0).astype(np.uint8) * 255
                    elif src_image.dtype == np.float32 or src_image.dtype == np.float64:
                        src_image_bin = (src_image > 0.1).astype(np.uint8) * 255
                    else:
                        # For uint8, uint16, etc.
                        try:
                            _, src_image_bin = cv2.threshold(src_image.astype(np.uint8), 1, 255, cv2.THRESH_BINARY)
                        except Exception as e:
                            # Fallback for problematic images
                            src_image_bin = (src_image > src_image.mean()/10).astype(np.uint8) * 255
                    
                    # Apply the 3-3-2 RGB "fire" colormap
                    h, w = src_image_bin.shape[:2]
                    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
                    
                    # Apply a custom 3-3-2 RGB LUT
                    # Red channel (3 bits) - 255
                    rgb_image[:,:,0] = (src_image_bin > 0) * 255
                    # Green channel (3 bits) - 210
                    rgb_image[:,:,1] = (src_image_bin > 0) * 210
                    # Blue channel (2 bits) - 150
                    rgb_image[:,:,2] = (src_image_bin > 0) * 150
                    
                    dic_rgb = rgb_image
                else:
                    dic_rgb = src_image
            else:
                # Process DIC image
                dic_rgb = dic_array
                # Convert to RGB if needed
                if len(dic_rgb.shape) == 2:
                    dic_rgb = cv2.cvtColor(auto_contrast(dic_rgb), cv2.COLOR_GRAY2RGB)
                else:
                    dic_rgb = auto_contrast(dic_rgb)
            
            # Use HTML and CSS for layered display with proper positioning
            # Convert images to base64 for embedding in HTML
            fluo_pil = Image.fromarray(fluo_rgb.astype('uint8'))
            fluo_buffer = io.BytesIO()
            fluo_pil.save(fluo_buffer, format="PNG")
            fluo_base64 = base64.b64encode(fluo_buffer.getvalue()).decode()
            
            # Calculate the transform for the DIC/mask image
            css_transform = f"translate({x_shift}px, {y_shift}px)"
            
            # Create HTML for layered display
            html = f"""
            <style>
                .overlay-container {{
                    position: relative;
                    width: 100%;
                    max-width: 800px;
                }}
                .base-image {{
                    display: block;
                    width: 100%;
                }}
                .overlay-image {{
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    opacity: {overlay_opacity/100};
                    transform: {css_transform};
                }}
            </style>
            <div class="overlay-container">
                <img src="data:image/png;base64,{fluo_base64}" class="base-image" />
            """
            
            # Add the DIC/mask image as overlay
            dic_pil = Image.fromarray(dic_rgb.astype('uint8'))
            dic_buffer = io.BytesIO()
            dic_pil.save(dic_buffer, format="PNG")
            dic_base64 = base64.b64encode(dic_buffer.getvalue()).decode()
            
            html += f"""
                <img src="data:image/png;base64,{dic_base64}" class="overlay-image" />
            </div>
            """
            
            # Display the layered images using HTML
            st.components.v1.html(html, height=fluo_rgb.shape[0] + 50)  # Add some margin
    
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
           - When satisfied with the alignment, use the download button to save the aligned overlay
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