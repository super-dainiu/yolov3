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
import config
from model import YOLOv3

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov3/ to path
colors = Colors()


class Detector(object):
    def __init__(self,
                 weights='pretrained/stage_2.pth.tar',  # model.pt path(s)
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
        img = letterbox(img0, self.imgsz)[0]
        img_ = img.copy()
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        img = img.unsqueeze(0)
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


def cv_imread(file_path):
    # OpenCV read image file for test
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    if cv_img.shape[2] == 4:
        return cv2.cvtColor(cv_img, cv2.COLOR_RGBA2RGB)
    return cv_img


def letterbox(im, new_shape=(config.IMAGE_SIZE, config.IMAGE_SIZE), color=(0, 0, 0), auto=False,
              scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


if __name__ == "__main__":
    img_dir = 'samples/football.jpg'
    my_img = cv_imread(img_dir)
    detector = Detector(weights='pretrained/stage_1.pth.tar',
                        conf_thres=0.8,
                        iou_thres=0.2,
                        view_time=True,
                        target='*')
    print(detector.target)
    my_img = detector.detection_image(my_img)
    cv2.imshow(img_dir, my_img)
    cv2.waitKey(0)
