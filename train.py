import config
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import random

from model import YOLOv3
from utils import (
    mean_average_precision,
    get_evaluation_bboxes,
    train_iter,
    test_iter,
)
from dataset import get_loaders
from detect import Detector
from torch_utils import (
    save_checkpoint,
    load_checkpoint,
)
from loss import YOLOLoss
import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True


def main():
    print(f'Using {config.DEVICE}')

    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY,
    )
    loss_fn = YOLOLoss()
    scaler = torch.cuda.amp.GradScaler()
    writer = SummaryWriter(config.WRITER)

    train_dataset, test_dataset, train_loader, test_loader = get_loaders(
        train_csv_path=config.DATASET+config.TRAIN_FILE, test_csv_path=config.DATASET+config.TEST_FILE
    )

    if config.LOAD_MODEL:
        load_checkpoint(config.PRETRAINED_FILE, model, optimizer, config.LEARNING_RATE,)

    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(2).repeat(1, 3, 2)
    ).to(config.DEVICE)

    for epoch in range(config.NUM_EPOCHS):
        print(f"Train epoch: {epoch + 1}")
        train_iter(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors, writer, epoch + config.CURRENT_EPOCH)
        print(f"Test epoch: {epoch + 1}")
        test_iter(test_loader, model, loss_fn, scaled_anchors, writer, epoch + config.CURRENT_EPOCH)

        if config.SAVE_MODEL:
            save_checkpoint(model, optimizer)

        if epoch % 1 == 0 and config.SAVE_MODEL:
            example, label = random.choice(test_dataset)
            detector = Detector(weights=config.CHECKPOINT_FILE,
                                return_img=True,
                                conf_thres=0.8,
                                iou_thres=0.2,
                                view_time=True,
                                target='*')
            detection, example = detector.detection_image(example)
            writer.add_image("results", example, epoch)

        model.train()

    pred_boxes, true_boxes = get_evaluation_bboxes(
        test_loader,
        model,
        iou_threshold=0.3,
        anchors=config.ANCHORS,
        threshold=0.7,
    )
    mapval = mean_average_precision(
        pred_boxes,
        true_boxes,
        iou_threshold=config.MAP_IOU_THRESH,
        box_format="midpoint",
        num_classes=config.NUM_CLASSES,
    )
    writer.add_scalars('map', {'map': mapval,
                               }, 0
                       )


if __name__ == "__main__":
    main()
