import streamlit as st
import main
import image_alignment

# Set page configuration
st.set_page_config(
    page_title="Yeast Cell Analysis Tools",
    page_icon="🔬",
    layout="wide"
)

# Dictionary of pages
PAGES = {
    "Cell Division Analysis": main,
    "Image Alignment Tool": image_alignment
}

# Sidebar for navigation
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

# Display the selected page
page = PAGES[selection]
page.app()