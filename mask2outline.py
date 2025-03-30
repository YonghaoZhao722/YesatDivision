import numpy as np
from skimage import measure
import tifffile as tiff
import os

# Path to your TIFF mask file
mask_path = r'/Users/zhaoyonghao/Documents/MATLAB/DIC_mask/WT+pWL74 CLB2_Q570 MS2v6_Q670_1_DIC_s1.tif'

# Output text file: same name as input but with .txt extension
output_file = os.path.splitext(mask_path)[0] + '.txt'

# Load the segmentation mask
mask = tiff.imread(mask_path)

# Find unique cell instance labels (excluding background = 0)
cell_labels = np.unique(mask)
cell_labels = cell_labels[cell_labels != 0]  # Remove background

# Store contours with their min Y-coordinate for sorting
cell_contours = []
for label in cell_labels:
    # Create a binary mask for the current cell
    binary_mask = (mask == label).astype(np.uint8)

    # Get the contours of the current cell
    contours = measure.find_contours(binary_mask, level=0.5)

    # Take the largest contour (assuming one main boundary per cell)
    contour = max(contours, key=len)  # Y, X coordinates

    # Get the minimum Y-coordinate for sorting (topmost point)
    min_y = np.min(contour[:, 0])

    cell_contours.append((label, contour, min_y))

# Sort cells by min Y-coordinate (top to bottom)
cell_contours = sorted(cell_contours, key=lambda x: x[2])  # Sort by min_y

# Open the output file for writing
with open(output_file, 'w') as fid:
    # Write FISH-QUANT header
    fid.write('FISH-QUANT\t\n')
    fid.write('File-version\t3D_v1\n')
    fid.write('RESULTS OF SPOT DETECTION PERFORMED ON 27-Mar-2025\n')
    fid.write('COMMENT\tAutomated outline definition from segmentation mask\n')
    fid.write('IMG_Raw\tMAX_WT+pWL74 CLB2_Q570 MS2v6_Q670_1_w2CY3-100-_s1.tif\n')
    fid.write('IMG_Filtered\t\n')
    fid.write('IMG_DAPI\t\n')
    fid.write('IMG_TS_label\t\n')
    fid.write('FILE_settings\t\n')
    fid.write('PARAMETERS\n')
    fid.write('Pix-XY\tPix-Z\tRI\tEx\tEm\tNA\tType\n')
    fid.write('160\t300\t1.33\t547\t583\t1.4\twidefield\n')

    # Process each cell instance in sorted order
    for i, (label, contour, min_y) in enumerate(cell_contours, 1):
        # Extract X and Y coordinates (swap because contours returns Y, X)
        y_coords = contour[:, 0]  # Y is row
        x_coords = contour[:, 1]  # X is column

        # Write CELL_START
        fid.write(f'CELL_START\tCell_{i}\n')

        # Write X_POS
        fid.write('X_POS')
        for x in x_coords:
            fid.write(f'\t{int(x)}')  # Convert to integer
        fid.write('\n')

        # Write Y_POS
        fid.write('Y_POS')
        for y in y_coords:
            fid.write(f'\t{int(y)}')  # Convert to integer
        fid.write('\n')

        # Write Z_POS (empty for 2D)
        fid.write('Z_POS\t\n')

        # Write CELL_END
        fid.write('CELL_END\n')

print(f'FISH-QUANT outline file saved as: {output_file}')