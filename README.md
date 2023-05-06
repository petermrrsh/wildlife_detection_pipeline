# wildlife_detection_pipeline

python3 ortho_tiling.py inputs/odm_orthophoto.tif
git clone https://github.com/WongKinYiu/yolov7.git
python3 yolov7/detect.py --weights inputs/weights.pt --conf 0.5 --source outputs/tiles --name labeled_tiles --save-txt
python3 process_labels.py inputs/odm_orthophoto.tif runs/detect labeled_tiles --tile-size 3040 4056