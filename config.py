import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'    # Avoid errors in Albumentation

DATASET = 'PASCAL_VOC'
CURRENT_EPOCH = 0
WRITER = 'runs'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 8
BATCH_SIZE = 4
IMAGE_SIZE = 416
NUM_CLASSES = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-4
NUM_EPOCHS = 100
CONF_THRESHOLD = 0.5
MAP_IOU_THRESH = 0.3
NMS_IOU_THRESH = 0.3
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]
PIN_MEMORY = True
LOAD_MODEL = True
SAVE_MODEL = True
PRETRAINED_FILE = "pretrained/yolov3.pth.tar"
CHECKPOINT_FILE = "checkpoint/checkpoint.pth.tar"
IMG_DIR = DATASET + "/images/"
LABEL_DIR = DATASET + "/labels/"
TRAIN_FILE = "/train.csv"
TEST_FILE = "/test.csv"
EXAMPLE = ["samples/football.jpg", "samples/bumblebee.jpg", "samples/rat.jpg"]

ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]  # From the YOLOv3 Paper

scale = 1.1

PASCAL_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]
