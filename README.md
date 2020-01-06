# Automatic-cell-detection-using-machine-learning
Predict the number of cells using deep learning. This work is done as a part of interview process.

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

# Preparing data for training 

1. Place training image inside yolov3/data/train/images

2. Place validation image inside yolov3/data/valid/images

3. Place training label inside yolov3/data/train/labels

4. Place validation label inside yolov3/data/valid/labels

5. Create medicalimage.txt which contains file paths of all train images and place it in folder yolov3/data/

6. Create medicalimagevalidation.txt which contains file paths of all validation images and place it in folder yolov3/data/

7. Create class.names file in the location yolov3/data and add the class names like below
class_0
class_1
class_2

8. Create medical.data file with the below information

classes=3 ( Specify number of classes present in the training dataset)

train=data/medicalimage.txt

valid=data/medicalimagevalidation.txt

names=data/class.names

# Starting the training

Run the script 

# python3 train.py 

Default training happens for 100 epochs with batch-size 4

if you would like to change those parameters using the below command line options


usage: train.py [-h] [--epochs EPOCHS] [--batch-size BATCH_SIZE]
                [--accumulate ACCUMULATE] [--cfg CFG] [--data DATA]
                [--multi-scale] [--img-size IMG_SIZE] [--rect] [--resume]
                [--transfer] [--nosave] [--notest] [--evolve]
                [--bucket BUCKET] [--cache-images] [--weights WEIGHTS]
                [--arc ARC] [--prebias] [--name NAME] [--device DEVICE]
                [--adam] [--var VAR]

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS
  --batch-size BATCH_SIZE
  --accumulate ACCUMULATE
                        batches to accumulate before optimizing
  --cfg CFG             *.cfg path
  --data DATA           *.data path
  --multi-scale         adjust (67%) img_size every 10 batches
  --img-size IMG_SIZE   inference size (pixels)
  --rect                rectangular training
  --resume              resume training from last.pt
  --transfer            transfer learning
  --nosave              only save final checkpoint
  --notest              only test final epoch
  --evolve              evolve hyperparameters
  --bucket BUCKET       gsutil bucket
  --cache-images        cache images for faster training
  --weights WEIGHTS     initial weights
  --arc ARC             yolo architecture
  --prebias             transfer-learn yolo biases prior to training
  --name NAME           renames results.txt to results_name.txt if supplied
  --device DEVICE       device id (i.e. 0 or 0,1 or cpu)
  --adam                use adam optimizer
  --var VAR             debug variable


                        

# The training procedure and testing procedure is modified from the below standard yolov3 pytorch implementation

https://github.com/ultralytics/yolov3
  
Sample training and validation data placed inside the file locations

![Train-data](https://github.com/VenkateshSatagopan/Automatic-cell-detection-using-machine-learning/tree/master/yolov3/data/train)

![Validation-data](https://github.com/VenkateshSatagopan/Automatic-cell-detection-using-machine-learning/tree/master/yolov3/data/train)


Sample output of the test-samples (Manually annoted and machine learning output) placed inside the folder


![Output of test-samples](https://github.com/VenkateshSatagopan/Automatic-cell-detection-using-machine-learning/tree/master/yolov3/data/samples/Sample-output-check)


# Future work

Reporting the image with the detected cell centers’ locations, ideally as an overlay on the original image. Currently working on to place the detected bounding box for each cell-type in the original image.




