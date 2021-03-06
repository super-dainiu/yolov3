import config
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import random

from model import YOLOv3
from utils import (
    train_iter,
    test_iter,
)
from dataset import get_loaders
from detect import Detector, imread
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
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS // 5)

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
        train_iter(train_loader, model, optimizer, loss_fn, scaler, scheduler, scaled_anchors, writer, epoch + config.CURRENT_EPOCH)
        print(f"Test epoch: {epoch + 1}")
        test_iter(test_loader, model, loss_fn, scaled_anchors, writer, epoch + config.CURRENT_EPOCH)
        scheduler.step()

        if (epoch + 1) % 10 == 0 and config.SAVE_MODEL:
            save_checkpoint(model, optimizer, config.CHECKPOINT_FILE.format(epoch + config.CURRENT_EPOCH))
            img0 = imread(random.choice(config.EXAMPLE))
            detector = Detector(weights=config.CHECKPOINT_FILE.format(epoch + config.CURRENT_EPOCH),
                                conf_thres=0.8,
                                iou_thres=0.2,
                                view_time=True,
                                target='*')
            img0 = detector.detection_image(img0)
            writer.add_image("results", img0.transpose(2, 0, 1), epoch + config.CURRENT_EPOCH)

        model.train()


if __name__ == "__main__":
    main()
