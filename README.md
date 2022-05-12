# YOLOv3 from scratch

This is the mid-term project for DATA130051 Computer Vision. 

<img src="random/yolov3.JPG"  />

YOLOv3 🚀 is an open-source object detection architecture first proposed by Joseph Redmon[[1](https://pjreddie.com/)]. I managed to reproduce the network from scratch with the help of Aladdin Persson[[2](https://github.com/aladdinpersson/)] and his excellent video tutorials[[3](https://www.youtube.com/watch?v=Grir6TZbc1M)]. Basically, I reimplemented YOLOv3 model as well as the utils and metrics for detections. Also, a detector class was written to provide a shortcut for video detections. Again, thanks a lot for those who contribute to this open-source society! 🤗

#### Install

[Python>=3.6.0](https://www.python.org/) is required with all [requirements.txt](./requirements.txt) installed including [PyTorch>=1.7](https://pytorch.org/get-started/locally/):

```bash
git clone https://github.com/super-dainiu/yolov3
cd yolov3
pip install -r requirements.txt
```

#### Download weights

Please follow the guide in [./pretrained](./pretrained) directory

#### Inference

After downloading the weights, you can do detection for images, gifs, or mp4 with [main.py](main.py).

>usage: main.py [-h] --weights WEIGHTS [--save SAVE] [--save_dir SAVE_DIR] [--conf_thres CONF_THRES] [--iou_thres IOU_THRES] [--max_det MAX_DET] [--target TARGET [TARGET ...]]
>          [--camera CAMERA] [--samples SAMPLES [SAMPLES ...]]
>
>Use YOLOv3
>
>optional arguments:
>
>-h, --help            show this help message and exit
>
>--weights WEIGHTS     Weight directory
>
>--save SAVE           Save results
>
>--save_dir SAVE_DIR   Save directory
>
>--conf_thres CONF_THRES
>                   Confidence threshold
>
>--iou_thres IOU_THRES
>                   IOU threshold
>
>--max_det MAX_DET     Maximum detection per frame
>
>--target TARGET [TARGET ...]
>                   Targets (i.e. person), * for all classes
>
>--camera CAMERA       Use your camera
>
>--samples SAMPLES [SAMPLES ...]
>                   Sample images (ends with .jpg, .png, .gif, .mp4)

You might do a simple detection with the following bash command.

```bash
python main.py --weights pretrained/yolov3.pth.tar --sample samples/video.mp4 samples/image_1.jpg samples/image_2.png samples/gif_1.gif --save True --save-dir outputs --target person car --conf_thres 0.7 --iou_thres 0.3 --max_det 10
```

Or activate your camera for detection.

```bash
python main.py --weights pretrained/yolov3.pth.tar --camera True
```

