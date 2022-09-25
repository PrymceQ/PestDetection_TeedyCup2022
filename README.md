# PestDetection_TeedyCup2022

Thanks to yolov5 https://github.com/ultralytics/yolov5 and https://github.com/open-mmlab/mmdetection.

We assemble two models - Faster RCNN and YOLOV5. They are in `./models`

1. `./Data_Format_Conversion` contains some data format conversion programs.

        `csv2coco.py`: csv data format to coco data format;
    
        `csv2pkl.py`: csv data format to pkl data format;
    
        `csv2yolo_txt.py`: csv data format to yolo(txt) data format;
    
        `pkl2csv.py`: pkl data format to csv data format.
    
2. `./Data_Visualization` contains some data visualization programs.

        `Invalid_target_filtering.py`: Some pictures have no label box, we distinguish them；
        
        `Visualization_of_average_pest_size.ipynb`: The average size of each pest is obtained according to the size of the pest label box；
        
        `Visualization_of_Detect_results.ipynb`: Visualization of target detection results.

3. `./tools` contains some useful tools.
     

