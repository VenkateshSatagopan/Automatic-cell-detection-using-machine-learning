import argparse
from sys import platform
from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
import numpy as np
import cv2
from torchvision import transforms
#import matplotlib.pyplot as plt


def detect(save_img=False):
    img_size = opt.img_size
    out, source, weights, half, view_img, save_txt = opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt
    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Initialize model
    model = Darknet(opt.cfg, img_size)
    # Load weights
    #attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)
    model.to(device).eval()
    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()
    save_img = True
    dataset = LoadImages(source, img_size=img_size, half=half)
    # Get names and colors
    names = load_classes(opt.names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    # Run inference
    for path, img, im0s, vid_cap in dataset:
        # Get detections
        img = torch.from_numpy(img).to(device)
        class_count = np.zeros(len( names))
        # Procedure added to split images into patches 416x416
        if opt.split_images or img.shape[1]>opt.img_size or img.shape[2]>opt.img_size:
          count=0
          remainder_shape_0 = img.shape[ 1 ] % opt.img_size
          remainder_shape_1 = img.shape[ 2 ] % opt.img_size

          if remainder_shape_1:
              remainder_shape_1 = opt.img_size - remainder_shape_1
          if remainder_shape_0:
              remainder_shape_0 = opt.img_size - remainder_shape_0

          padded_img_width = img.shape[ 1 ] + remainder_shape_0
          padded_img_height = img.shape[ 2 ] + remainder_shape_1


          padded_image = torch.zeros((img.shape[ 0 ], padded_img_width, padded_img_height))
          padded_image[:,:img.shape[1],:img.shape[2]]=img

          for i in range( 0, padded_image.shape[1], opt.img_size):
            for j in range( 0, padded_image.shape[2], opt.img_size):
                count+=1
                class_count=prediction(model,padded_image[:,i:i+opt.img_size,j:j+opt.img_size], path, im0s, out, names, save_img,colors,class_count,count)
                #print(class_count)
        else:
          class_count=prediction(model,img,path,im0s,out,names,save_img,colors,class_count)

        class_count=np.array(class_count)
        for i in range( class_count.shape[0]):
            print( "Total-class-count of class " + str( i ) + " is " + str(class_count[i]))



def prediction(model,img,path,im0s,out,names,save_img,colors,class_count_array,count=0):

        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        img_1 = transforms.ToPILImage()(img[ 0, :, :, : ]).convert( "RGB" )
        img_1=np.asarray(np.uint8(img_1))

        t = time.time()

        pred = model(img)[0]


        if opt.half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes)



        # Process detections

        for i, det in enumerate(pred):  # detections per image

            p, s, im0 = path, '', im0s

            if count:
              save_path = str(Path(out))+'/'+str(count)+str(Path(p).name)
            else:
              save_path = str(Path(out) / Path( p ).name )


            #s += '%gx%g ' % img.shape[2:]  # print string

            s  +='\nPatch '
            s  += '%g ' % count
            s  += 'Output: '



            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()


                for c in det[:, -1].unique():

                    n = (det[:, -1] == c).sum()  # detections per class
                    class_count_array[int(c)]=class_count_array[int(c)]+n.cpu().detach().numpy()
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    if opt.save_txt:  # Write to file
                        with open(save_path + '.txt', 'a') as file:
                            file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))
                    if save_img or opt.view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

            print('%sDone. (%.3fs)' % (s, time.time() - t))
            if save_img:
                cv2.imwrite(save_path,im0)
        return class_count_array








if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-tiny.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/class.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='weights/last.pt', help='path to weights file')
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.6, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='display results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--split-images',type=bool,default=False,help='set the parameter to True if the image size > 416')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect()
