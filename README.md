# Automatic-cell-detection-using-machine-learning
Predict the number of cells using deep learning

There are three types of cells in the given image. Each image has three types of cells . This project aims to predict cell count for each type of cell using modified version of yolov3.

# Cell -type 0
![Class-0](https://github.com/VenkateshSatagopan/Automatic-cell-detection-using-machine-learning/blob/master/cell-types/cell_type_0.png)


# Cell-type 1
![Class-1](https://github.com/VenkateshSatagopan/Automatic-cell-detection-using-machine-learning/blob/master/cell-types/cell_type_1.png)

# Cell-type 2
![Class-0](https://github.com/VenkateshSatagopan/Automatic-cell-detection-using-machine-learning/blob/master/cell-types/cell_type_2.png)


The project aims to find number of each cell types inside the tissue region(gray background region) from the image like shown below
# Bigger image
![Bigger-image](https://github.com/VenkateshSatagopan/Automatic-cell-detection-using-machine-learning/blob/master/cell-types/black_bubbles_1.png)



# Steps for Testing the image

Use the script detect.py

# python3 detect.py -h
usage: detect.py [-h] [--cfg CFG] [--names NAMES] [--weights WEIGHTS]
                 [--source SOURCE] [--output OUTPUT] [--img-size IMG_SIZE]
                 [--conf-thres CONF_THRES] [--iou-thres IOU_THRES]
                 [--fourcc FOURCC] [--half] [--device DEVICE] [--view-img]
                 [--save-txt] [--classes CLASSES [CLASSES ...]]
                 [--split-images SPLIT_IMAGES]

optional arguments:
  -h, --help            show this help message and exit
  --cfg CFG             *.cfg path
  --names NAMES         *.names path
  --weights WEIGHTS     path to weights file
  --source SOURCE       source
  --output OUTPUT       output folder
  --img-size IMG_SIZE   inference size (pixels)
  --conf-thres CONF_THRES
                        object confidence threshold
  --iou-thres IOU_THRES
                        IOU threshold for NMS
  --fourcc FOURCC       output video codec (verify ffmpeg support)
  --half                half precision FP16 inference
  --device DEVICE       device id (i.e. 0 or 0,1) or cpu
  --view-img            display results
  --save-txt            display results
  --classes CLASSES [CLASSES ...]
                        filter by class
  --split-images SPLIT_IMAGES
                        set the parameter to True if the image size > 416



Directly place the test images inside yolov3/data/samples and run the script
# python3 detect.py
or specify the location of test images using command line options and run the above script

