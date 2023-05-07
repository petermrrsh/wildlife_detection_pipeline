from PIL import Image
from osgeo import gdal
import os
import shutil
import sys
import cv2
from matplotlib import pyplot as plt

def parse_args():
    if len(sys.argv) < 2 or len(sys.argv) > 5:
        print("Usage: ortho_tiling.py <filename> [--tile-size <tile_width> <tile_height>]")
        sys.exit(1)

    filename = sys.argv[1]

    tile_width = None
    tile_height = None

    if len(sys.argv) == 5:
        if sys.argv[2] != "--tile-size":
            print("Usage: ortho_tiling.py <filename> [--tilesize <tile_width> <tile_height>]")
            sys.exit(1)
        try:
            tile_width = int(sys.argv[3])
            tile_height = int(sys.argv[4])
        except ValueError:
            print("Tile width and height must be integers")
            sys.exit(1)

    return filename, tile_width, tile_height

def plot_rectangle(x, y, width, height, img, line_thickness=3):
    color = [0, 0, 255]
    p1, p2 = (x, y), (x + width, y + height)
    cv2.rectangle(img, p1, p2, color, thickness=int(line_thickness), lineType=cv2.LINE_AA)

filename, tile_width, tile_height = parse_args()

# Define the size of the sub-images
tile_width = tile_width or 4056
tile_height = tile_height or 3040

# Set the overlap between tiles, must be at least double the size of whatever you're trying to identify
overlap_x = 500
overlap_y = 500

# Define the stride of the sliding window
stride_x = tile_width - overlap_x
stride_y = tile_height - overlap_y

print(f"Opening input file: '{filename}'")

# Open the orthophoto with gdal for segmentation
ds = gdal.Open(filename)

if ds is None:
    print(f"Failed to open '{filename}' with gdal.")
    sys.exit(1)

# Open the orthophoto with cv2 for display
display_img = cv2.imread(filename)

if display_img is None:
    print(f"Failed to read '{filename}' with cv2.")
    sys.exit(1)

# Name the output folders
out_dir = "outputs"
tile_dir = f"{out_dir}/tiles"

# Create the output folder(s)
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
if os.path.exists(tile_dir):
    shutil.rmtree(tile_dir)
os.mkdir(tile_dir)

# Create a directory for the gdal translate file
if not os.path.exists("gdal"):
    os.mkdir("gdal")

# Get the size of the image
width = ds.RasterXSize
height = ds.RasterYSize

# Print input image size and tile size
print(f"\nInput size: {height} x {width}")
print(f"Tile size:  {tile_height} x {tile_width}\n")

# Iterate over tiles (sub-images) using the sliding window approach
for x in range(0, width, stride_x):
    for y in range(0, height, stride_y):
        
        # Check if we're at the right edge of the image
        if x + tile_width > width:
            x = width - tile_width
            
        # Check if we're at the bottom edge of the image
        if y + tile_height > height:
            y = height - tile_height
            
        # Plot each tile for display purposes
        plot_rectangle(x, y, tile_width, tile_height, display_img, 0.002 * width)
        
        try:
            sub_ds = gdal.Translate('gdal/trans', ds, format="GTiff", srcWin=[x, y, tile_width, tile_height])
            sub_img = sub_ds.ReadAsArray()
            sub_ds = None
            
            # Convert the sub-image to a PIL image
            sub_img = sub_img.transpose(1, 2, 0)
            sub_img = Image.fromarray(sub_img).convert("RGB")
            
            # Save the sub-image as a JPEG file
            tile_name = f'{x}_{y}_tile.jpg'
            sub_img.save(f'{tile_dir}/{tile_name}')
            print(f"Saving tile: '{tile_name}'")
            
        except Exception as e:
            print(f"Error processing tile at x={x}, y={y}: {str(e)}")
            continue

out_image_path = f"{out_dir}/display_img.tif"
print(f"\nSaving display img at '{out_image_path}'")
cv2.imwrite(out_image_path, display_img)

plt.imshow(display_img)
print("Done")

