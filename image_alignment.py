import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import os
import tifffile
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
                
                # Convert to binary mask if needed
                _, mask_binary = cv2.threshold(mask_array, 1, 255, cv2.THRESH_BINARY)
                
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
        
        # Create sliders for shifting
        col1, col2 = st.columns(2)
        with col1:
            x_shift = st.slider("Shift X (horizontal)", -width//2, width//2, 0)
        with col2:
            y_shift = st.slider("Shift Y (vertical)", -height//2, height//2, 0)
            
        # Overlay opacity control
        overlay_opacity = st.slider(
            "Overlay Opacity (%)", 
            min_value=10, 
            max_value=100, 
            value=50,
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
                # Convert to binary first
                _, src_image_bin = cv2.threshold(src_image, 1, 255, cv2.THRESH_BINARY)
                
                # Apply color
                cmap = plt.colormaps['plasma']
                src_image_color = cmap(src_image_bin.astype(float) / 255.0)
                src_image_color = (src_image_color[:, :, :3] * 255).astype(np.uint8)
                src_image = src_image_color
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
        
        # Create overlay
        alpha = overlay_opacity / 100.0
        overlay = cv2.addWeighted(shifted_image, alpha, fluo_rgb, 1.0 - alpha, 0)
        
        # Display the overlay
        st.image(overlay, caption=f"Aligned Overlay (Shift X: {x_shift}, Y: {y_shift})", use_container_width=True)
        
        # Display the current alignment values
        st.success(f"Current alignment: X shift = {x_shift} pixels, Y shift = {y_shift} pixels")
        
        # Add download button for the aligned overlay
        buf = io.BytesIO()
        plt.imsave(buf, overlay)
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
        4. Adjust the overlay opacity to best visualize the alignment
        5. Download the aligned overlay image when satisfied
        
        ### Tips:
        
        - For multichannel fluorescence images, you can select specific channels
        - You can choose to overlay either the DIC image or the segmentation mask on the fluorescence image
        - The shift values (X and Y) tell you how much the images needed to be aligned, which can be useful for batch processing
        """)

if __name__ == "__main__":
    app()