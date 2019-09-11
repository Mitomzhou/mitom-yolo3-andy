'''
Created on Jul 24, 2019

@author: monky
'''
import sys
import time
from PIL import Image, ImageDraw
import cv2
#from models.tiny_yolo import TinyYoloNet
from utils import *
from image import letterbox_image, correct_yolo_boxes
from darknet import DarkNet


def detect(cfgfile, weightfile, imgfile):
    m = DarkNet(cfgfile)

    #m.print_net()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if m.num_classes == 20:
        namesfile = '../data/voc.names'
    elif m.num_classes == 80:
        namesfile = '../data/coco.names'
    else:
        namesfile = '../data/names'
    
    img = Image.open(imgfile).convert('RGB')
    sized = letterbox_image(img, m.width, m.height)

    start = time.time()
    
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        m.cuda()
    
    boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
    correct_yolo_boxes(boxes, img.width, img.height, m.width, m.height)

    finish = time.time()
    print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

    class_names = load_class_names(namesfile)
    plot_boxes(img, boxes, 'predictions.jpg', class_names)
    predictions_img = Image.open('predictions.jpg')
    predictions_img.show()
    
    
def camera_detect(cfgfile, weightfile):
    """
    - camera detect
    :cfgfile    use tiny config file
    :weightfile use tiny weight file 
    """
    model = DarkNet(cfgfile)
    model.print_net()
    model.load_weights(weightfile)
    print('load weights done!')
    
    
    num_classes = 80
    if num_classes == 20:
        namesfile = '../data/voc.names'
    elif num_classes == 80:
        namesfile = '../data/coco.names'
    else:
        namesfile = '../data/names'
    
    cap = cv2.VideoCapture(0)  
    if not cap.isOpened():
        print("Unable to open camera")
        exit(-1)
    while True:
        res, img = cap.read()
        if res:
            img = Image.fromarray(img, mode='RGB') # numpy.array -> PIL.Image
            sized = letterbox_image(img, model.width, model.height)

            boxes = do_detect(model, sized, 0.5, 0.4, False)
            correct_yolo_boxes(boxes, img.width, img.height, model.width, model.height)
            class_names = load_class_names(namesfile)
            image_draw = plot_boxes(img, boxes, None, class_names)
            
            np_img = np.asarray(image_draw)   # PIL.Image -> numpy.array  
            cv2.imshow(cfgfile, np_img)
            cv2.waitKey(1)
        else:
            print("Unable to read image")
            exit(-1) 
    
if __name__ == '__main__':
    detect('../cfg/yolov3.cfg', '../weight/yolov3.weights', '../data/stree.jpeg')
#     detect('../cfg/yolov3-tiny.cfg', '../weight/yolov3-tiny.weights', '../data/stree.jpeg')
#     camera_detect('../cfg/yolov3.cfg', '../weight/yolov3.weights')
    
    
    