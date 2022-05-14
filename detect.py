import sys
import os
from pathlib import Path
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import torch
import numpy as np
from utils import non_max_suppression, cells_to_bboxes
from torch_utils import attempt_load, time_synchronized
from plots import Colors, plot_boxes
from augmentations import detect_transforms, image_transforms
import config
from model import YOLOv3
from PIL import Image

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov3/ to path
colors = Colors()


class Detector(object):
    def __init__(self,
                 weights='pretrained/yolov3-250.pth.tar',  # model.pt path(s)
                 imgsz=416,  # inference size (pixels)
                 conf_thres=0.75,  # confidence threshold
                 iou_thres=0.6,  # NMS IOU threshold
                 max_det=1000,  # maximum detections per image
                 opt_device=config.DEVICE,  # cuda device, i.e. 0 or 0,1,2,3 or cpu,
                 view_time=False,  # show time
                 half=False,  # use FP16 half-precision inference
                 target=("*", ),  # targets for detection
                 ):

        # detection configs
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.imgsz = imgsz
        self.target = target if '*' not in target else None  # '*' to detect all possible classes

        # return configs
        self.view_time = view_time

        # load model
        self.model = YOLOv3().to(opt_device)  # load FP32 model
        self.device = opt_device
        attempt_load(weights, self.model)
        self.half = half & (opt_device != 'cpu')
        if self.half:  # half precision only supported on CUDA
            self.model.half()

        # load class names
        self.names = config.PASCAL_CLASSES
        if not self.target:
            self.target = self.names

        # run once
        if self.device != 'cpu':
            self.model(torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(self.model.parameters())))

    def detection_image(self, img0):

        # img0 to tensor
        t0 = time_synchronized()
        img = detect_transforms(image=img0)['image']
        img_ = image_transforms(image=img0)['image']
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img = img.unsqueeze(0).to(config.DEVICE)
        t0 = time_synchronized()-t0

        # Do detections
        t1 = time_synchronized()

        pred = self.model(img)
        bboxes = [[]]
        for i in range(3):
            S = pred[i].shape[2]
            anchor = torch.tensor([*config.ANCHORS[i]]).to(self.device) * S
            boxes_scale_i = cells_to_bboxes(
                pred[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box
        bboxes = non_max_suppression(bboxes[0],
                                     self.iou_thres,
                                     self.conf_thres,
                                     max_det=self.max_det)

        t1 = time_synchronized()-t1

        # Process detections
        t2 = time_synchronized()
        img_ = plot_boxes(img_, bboxes, targets=self.target)
        t2 = time_synchronized()-t2

        # Print time
        if self.view_time:
            print(f'Detection:\n'
                  f'Pre-process consumption: {t0:.3f}s\n'
                  f'Detection consumption: {t1:.3f}s'
                  f'\nProcess consumption: {t2:.3f}s.\n')

        return np.ascontiguousarray(img_)


def imread(file_path):
    # OpenCV read image file for test
    image = np.array(Image.open(file_path).convert("RGB"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


if __name__ == "__main__":
    img_dir = 'samples/football.jpg'
    my_img = imread(img_dir)
    detector = Detector(weights='pretrained/yolov3-250.pth.tar',
                        conf_thres=0.8,
                        iou_thres=0.2,
                        view_time=True,
                        target='*')
    print(detector.target)
    my_img = detector.detection_image(my_img)
    cv2.imshow(img_dir, my_img)
    cv2.waitKey(0)
