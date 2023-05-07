import json
import argparse
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser(prog="kml_creator.py", description="Parse arguments for kml creator")
    parser.add_argument("json_file", type=str, help="Json labels filepath (i.e. 'outputs/gps_labels.json')")
    args = parser.parse_args()

    # Retrieve parsed arguments
    json_file = args.json_file

    return json_file

# Read in labels from the json file
def read_labels_from_json(json_file):
    with open(json_file, 'r') as f:
        bounding_boxes = json.load(f)
    return bounding_boxes

json_file = parse_args()

animal = "Cow"
display_name = f"{animal} Detections"
output_filename = f"outputs/{animal.lower()}_detections.kml"
print("Generating KML for Google Earth")
print(f"  Animal:       {animal}")
print(f"  Name:         '{display_name}'")
print(f"  Input (json): '{json_file}'")
print(f"  Output (kml): '{output_filename}'")

# Read in labels from the json file
try:
    coords = read_labels_from_json(json_file)
except:
    print(f"Error: the json file '{json_file}' could not be read.")
    sys.exit(1)

# Clear the output file if there is one and then open it in append mode
if os.path.exists(output_filename):
    with open(output_filename, "w") as file:
        file.truncate(0)
output_file = open(output_filename, "a")

# generate kml for the header and bounding box style
print(
'<?xml version="1.0" encoding="UTF-8"?>\n'
'<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2" xmlns:kml="http://www.opengis.net/kml/2.2" xmlns:atom="http://www.w3.org/2005/Atom">\n'
'<Document>\n'
f'   <name>{display_name}</name>\n'
'	<Style id="bounding_box">\n'
'       <LineStyle>\n'
'           <width>1.0</width>\n'
'		    <color>ff0000ff</color>\n'
'       </LineStyle>\n'
'       <PolyStyle>\n'
'		    <fill>0</fill>\n'
'       </PolyStyle>\n'
'   </Style>\n\n', file=output_file)

# constants for indexing into coordinates
X_MIN = 0
Y_MIN = 1
X_MAX = 2
Y_MAX = 3

# generate kml for each bounding box
for coord in coords:
    
    # Bounding Box
    # coord_A ------------------ coord_B
    # |                                |
    # |                                |
    # |                                |
    # |                                |
    # coord_D ------------------ coord_C
    
    coord_A = [coord[X_MIN], coord[Y_MIN]]
    coord_B = [coord[X_MAX], coord[Y_MIN]]
    coord_C = [coord[X_MIN], coord[Y_MAX]]
    coord_D = [coord[X_MAX], coord[Y_MAX]]

    print(
    '<Placemark>\n'
    f'	<name>{animal.lower()}</name>\n'
    '	<styleUrl>#bounding_box</styleUrl>\n'
    '	<Polygon>\n'
    '   	<tessellate>1</tessellate>\n'
    '	        <outerBoundaryIs>\n'
    '		        <LinearRing>\n'
    '				    <coordinates>\n'
    f'						{coord_A[1]},{coord_A[0]},0 {coord_B[1]},{coord_B[0]},0 {coord_D[1]},{coord_D[0]},0 {coord_C[1]},{coord_C[0]},0 {coord_A[1]},{coord_A[0]},0\n'
    '					</coordinates>\n'
    '			    </LinearRing>\n'
    '			</outerBoundaryIs>\n'
    '	</Polygon>\n'
    '</Placemark>\n', file=output_file)

# generate kml for the footer
print(
'\n</Document>\n'
'</kml>\n', file=output_file)

# close the output file
output_file.close()
print(f"\n{len(coords)} labels written to '{output_filename}'")