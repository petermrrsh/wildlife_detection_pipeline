# wildlife_detection_pipeline

Copy the orthophoto.tif file into inputs

python3 ortho_tiling.py inputs/odm_orthophoto.tif
git clone https://github.com/WongKinYiu/yolov7.git
python3 yolov7/detect.py --weights inputs/example_weights.pt --conf 0.5 --source outputs/tiles --name labeled_tiles --save-txt
python3 process_labels.py inputs/odm_orthophoto.tif runs/detect labeled_tiles
python3 kml_creator.py outputs/gps_labels.json