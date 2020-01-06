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



# Step 1:
place the test images inside yolov3/data/samples
# Step 2:
Run the script 
# python3 detect.py

# Steps for training the image

# Step 1
# Creating the training dataset patches
For creating the patches, run the script
# python3 create_patches.py

usage: create-patches.py [-h] [--input-folder INPUT_FOLDER]
                         [--output-folder OUTPUT_FOLDER]
                         [--patch-size PATCH_SIZE] [--num-patches NUM_PATCHES]

optional arguments:
  -h, --help            show this help message and exit
  --input-folder INPUT_FOLDER
                        source
  --output-folder OUTPUT_FOLDER
                        output folder
  --patch-size PATCH_SIZE
                        inference size (pixels)
  --num-patches NUM_PATCHES
                        Number of patches to be extracted
                        
                        

Before starting the script, put the training images inside folder Input/ or specify the input-folder location using command line options in the above script.

# Step 2

For annotation of patches

I have used CVAT toolbox to annotate the training patches created from step 1.
https://github.com/opencv/cvat

Labels can be obtained using the above tool for training and validation.

# Step 3

Preparing data for training 

Place training image inside data/train/images

Place validation image inside data/valid/images

Place training label inside data/train/labels

Place validation label inside data/valid/labels

                        
                        
  




