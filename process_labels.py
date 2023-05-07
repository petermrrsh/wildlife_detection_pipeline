import os
import random
import cv2
from osgeo import gdal, osr
import json
import argparse
import glob
import sys
from matplotlib import pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(prog="process_labels.py", description="Parse arguments for label processor")

    # Positional arguments
    parser.add_argument("input_filename", type=str, help="Input filename (i.e. 'inputs/orthophoto.tif')")
    parser.add_argument("yolo_run_path", type=str, help="YOLO run path (i.e. 'runs/detect')")
    parser.add_argument("yolo_run_name", type=str, help="YOLO run name (i.e. 'labeled_tiles')")

    args = parser.parse_args()

    # Retrieve parsed arguments
    input_filename = args.input_filename
    yolo_run_path = args.yolo_run_path
    yolo_run_name = args.yolo_run_name

    return input_filename, yolo_run_path, yolo_run_name

def find_highest_numbered_file(directory, prefix):
    max_number = -1
    max_file = None
    
    for filename in os.listdir(directory):
        if filename.startswith(prefix) and filename[len(prefix):].isdigit():
            number = int(filename[len(prefix):])
            if number > max_number:
                max_number = number
                max_file = filename
    
    if max_file is not None:
        return os.path.join(directory, max_file)
    else:
        return None
    
def string_to_numeric_list(str):
    result = []
    for part in str.split(" "):
        result.append(float(part))
    return result
    
def list_txt_files_in_dir(dir_path):
    file_list = []
    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)
        if os.path.isfile(file_path) and file.lower().endswith(".txt"):
            file_list.append(file_path)
    return file_list

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        

def parse_label(label):
    bbox = label.split(" ")[1:]
    
    CENTER_X = 0
    CENTER_Y = 1
    WIDTH = 2
    HEIGHT = 3
    
    try:
        return float(bbox[CENTER_X]), float(bbox[CENTER_Y]), float(bbox[WIDTH]), float(bbox[HEIGHT])
    # returns none if label is invalid
    except (ValueError, IndexError):
        return None
    
# Calculate the IOU (intersection over union) for two bounding boxes
def calculate_iou(box1, box2):
    # Extract box coordinates
    x1_box1, y1_box1, x2_box1, y2_box1 = box1
    x1_box2, y1_box2, x2_box2, y2_box2 = box2

    # Calculate coordinates of the intersection rectangle
    x_intersection = max(x1_box1, x1_box2)
    y_intersection = max(y1_box1, y1_box2)
    w_intersection = max(0, min(x2_box1, x2_box2) - x_intersection)
    h_intersection = max(0, min(y2_box1, y2_box2) - y_intersection)

    # Calculate area of intersection rectangle
    area_intersection = w_intersection * h_intersection

    # Calculate area of union
    area_box1 = (x2_box1 - x1_box1) * (y2_box1 - y1_box1)
    area_box2 = (x2_box2 - x1_box2) * (y2_box2 - y1_box2)
    area_union = area_box1 + area_box2 - area_intersection

    # Calculate IoU
    iou = area_intersection / area_union

    return iou

# Return true if labels overlap, false otherwise
def labels_overlap(bbox1, bbox2, overlap_threshold):
    if calculate_iou(bbox1, bbox2) > overlap_threshold:
        return True
    else:
        return False
    
# Return the max combination of two bounding boxes
def combine_labels(label1, label2):
    x1_label1, y1_label1, x2_label1, y2_label1 = label1
    x1_label2, y1_label2, x2_label2, y2_label2 = label2
    
    # Find the minimum and maximum coordinates
    x_min = min(x1_label1, x1_label2)
    y_min = min(y1_label1, y1_label2)
    x_max = max(x2_label1, x2_label2)
    y_max = max(y2_label1, y2_label2)
    
    # Return the combined label
    return [x_min, y_min, x_max, y_max]

# Using geo transform data from the .tif file, translate an x and y pixel into gps space
def pixel_to_gps_coordinates(dataset, x_pixel, y_pixel):

    # Get the GeoTransform information
    geo_transform = dataset.GetGeoTransform()

    # Extract the necessary information
    origin_x = geo_transform[0]
    origin_y = geo_transform[3]
    pixel_width = geo_transform[1]
    pixel_height = geo_transform[5]

    # Calculate the GPS coordinates
    gps_x = origin_x + x_pixel * pixel_width
    gps_y = origin_y + y_pixel * pixel_height

    # Create a spatial reference object for the GPS coordinates
    srs = osr.SpatialReference()
    srs.ImportFromWkt(dataset.GetProjection())

    # Transform the GPS coordinates to WGS84 (Used by Google Earth)
    target_srs = osr.SpatialReference()
    target_srs.SetWellKnownGeogCS("WGS84")

    transform = osr.CoordinateTransformation(srs, target_srs)
    gps_x, gps_y, _ = transform.TransformPoint(gps_x, gps_y)

    return gps_x, gps_y

# Generate a simple json file to store all the labels
def write_labels_to_json(bounding_boxes, json_file):
    with open(json_file, 'w') as f:
        json.dump(bounding_boxes, f)

input_filename, yolo_run_path, yolo_run_name = parse_args()

# Make sure there is an outputs folder and set output filenames
output_dir = "outputs"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
save_path = f"{output_dir}/orthophoto_labeled.tif"
gps_labels_json = f"{output_dir}/gps_labels.json"

# Find the latest run of yolov7 object detection
print(f"Searching '{yolo_run_path}' for latest run of '{yolo_run_name}'")
detect_dir = find_highest_numbered_file(yolo_run_path, yolo_run_name)
if detect_dir is None:
    print(f"Error: No runs named '{yolo_run_name}' were found at '{yolo_run_path}'. "
          "Make sure the path is correct and the name matches your last yolo run.")
    sys.exit(1)
print(f"Found: '{detect_dir}'")

# Make sure the label folder is present
label_dir = detect_dir + "/labels"
if not os.path.exists(label_dir):
    print(f"Error: No labels were found in {detect_dir}. Make sure YOLOv7 is run with --save-txt.")
    sys.exit(1)

# Get the width and height of a tile from the latest run
try:
    first_tile = glob.glob(detect_dir + "/*.jpg")[0]
except:
    print(f"No labeled images were found in {detect_dir}")
    sys.exit(1)
tile = cv2.imread(first_tile)
tile_height, tile_width, _ = tile.shape
print(f"Tile size: {tile_height} x {tile_width}")

# Open the input file (.tif) with cv2 and gdal (both are needed)
print(f"Opening input file: '{input_filename}'")
ortho_image = cv2.imread(input_filename) 
assert ortho_image is not None, f"Could not open '{input_filename}' with cv2"
ortho_image_gdal = gdal.Open(input_filename)
assert ortho_image is not None, f"Could not open '{input_filename}' with gdal"

# Get the input image's height and width
ortho_height, ortho_width, _ = ortho_image.shape

# Calculate each label's pixel coordinates in context of the larger image and append to labels_in_context
print("\nReading in labels")
labels_in_context = []
for filename in list_txt_files_in_dir(label_dir):
    
    # get the x_offset and the y_offset from the filename
    filename_parts = os.path.basename(filename).split("_")
    if len(filename_parts) < 2 or not (filename_parts[0].isdigit() and filename_parts[1].isdigit()):
        continue
    x_offset = int(filename_parts[0])
    y_offset = int(filename_parts[1])
    
    with open(filename, "r") as file:
        labels = file.readlines()
        
        for label in labels:
            # read in a label's x, y, width, height coordinates
            x_center_val, y_center_val, width_val, height_val = parse_label(label)
            if x_center_val is None:
                continue
            
            # calculate their position within the larger image
            x_center = x_offset + (tile_width * x_center_val)
            y_center = y_offset + (tile_height * y_center_val)
            width = tile_width * width_val
            height = tile_height * height_val
            
            # format coordinates as x1, y1, x2, y2, and add to labels_in_context
            x1 = round(x_center - (width/2))
            y1 = round(y_center - (height/2))
            x2 = round(x_center + (width/2))
            y2 = round(y_center + (height/2))
            labels_in_context.append([x1, y1, x2, y2])
print(f"Done, {len(labels_in_context)} labels found")

# Loop over all labels_in_context and combine overlapping labels
overlap_threshold = 0.6
print(f"\nCombining overlaps (overlap_threshold = {overlap_threshold})")
skip = []
for i in range(len(labels_in_context)):
    for j in range(i + 1, len(labels_in_context)):
        
        # Combine each set of labels whose IOU is greater than the overlap_threshold
        if labels_overlap(labels_in_context[i], labels_in_context[j], overlap_threshold):
            
            labels_in_context[j] = combine_labels(labels_in_context[i], labels_in_context[j])
            
            # Note which labels can be skipped in the next step
            skip.append(i)
print(f"Done, {len(skip)} labels combined")

# Loop over the combined labels, plot them on the larger image and record their gps coordinates
gps_labels = []
for i in range(len(labels_in_context)):
    if i in skip:
        continue
    
    # plot labels on the output image (for display)
    plot_one_box(labels_in_context[i], ortho_image, label="Cow", color=[255, 0, 0], line_thickness=3)
    x1, y1, x2, y2 = labels_in_context[i]
    
    # calculate label's gps coordinates using gdal
    gps_x1, gps_y1 = pixel_to_gps_coordinates(ortho_image_gdal, x1, y1)
    gps_x2, gps_y2 = pixel_to_gps_coordinates(ortho_image_gdal, x2, y2)

    gps_labels.append([gps_x1, gps_y1, gps_x2, gps_y2])

# save the gps labels in a json file so they can be read by kml_creator.py
print(f"\nGPS labels saved in: '{gps_labels_json}'")
write_labels_to_json(gps_labels, gps_labels_json)

# save the labeled .tif for viewing purposes
cv2.imwrite(save_path, ortho_image)
plt.imshow(ortho_image)

print(f"The image showing plotted labels is saved in: '{save_path}'")
