import csv
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser(prog="kml_creator.py", description="Parse arguments for kml creator")
    parser.add_argument("json_file", type=str, help="Json labels filepath (i.e. 'outputs/gps_labels.json')")
    args = parser.parse_args()

    # Retrieve parsed arguments
    json_file = args.json_file

    return json_file

def write_coords_to_csv(coords, filename):
    fieldnames = ['x_min', 'y_min', 'x_max', 'y_max']

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for coord in coords:
            writer.writerow({'x_min': coord[0], 'y_min': coord[1], 'x_max': coord[2], 'y_max': coord[3]})
            
# Read in labels from the json file
def read_labels_from_json(json_file):
    with open(json_file, 'r') as f:
        bounding_boxes = json.load(f)
    return bounding_boxes

animal = "Cow"
output_filename = f"outputs/{animal.lower()}_detections.csv"

json_file = parse_args()

print("Generating CSV")
print(f"  Input (json): '{json_file}'")
print(f"  Output (kml): '{output_filename}'")

coords = read_labels_from_json(json_file)
write_coords_to_csv(coords, output_filename)