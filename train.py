import config
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os

from model import YOLOv3
from tqdm import tqdm
from utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples
)
from loss import YOLOLoss
import warnings
warnings.filterwarnings("ignore")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

torch.backends.cudnn.benchmark = True


def train(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors, writer, epoch):
    loop = tqdm(train_loader, leave=True)
    losses = []

    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )

        with torch.cuda.amp.autocast():
            out = model(x)
            loss = (
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2])
            )

        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)
        writer.add_scalar('train_loss', mean_loss, epoch * len(loop) + batch_idx)


def main(writer):
    print(f'Using {config.DEVICE}')
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY,
    )
    loss_fn = YOLOLoss()
    scaler = torch.cuda.amp.GradScaler()

    train_loader, test_loader, train_eval_loader = get_loaders(
        train_csv_path=config.DATASET+config.TRAIN_FILE, test_csv_path=config.DATASET+config.TEST_FILE
    )

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE,)

    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(2).repeat(1, 3, 2)
    ).to(config.DEVICE)

    for epoch in range(config.NUM_EPOCHS):
        train(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors, writer, epoch + config.CURRENT_EPOCH)
        if config.SAVE_MODEL:
            save_checkpoint(model, optimizer)

        class_acc_train, noobj_acc_train, obj_acc_train = check_class_accuracy(model, train_loader, threshold=config.CONF_THRESHOLD)
        class_acc_test, noobj_acc_test, obj_acc_test = check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)
        pred_boxes, true_boxes = get_evaluation_bboxes(
            test_loader,
            model,
            iou_threshold=config.NMS_IOU_THRESH,
            anchors=config.ANCHORS,
            threshold=config.CONF_THRESHOLD,
        )
        mapval = mean_average_precision(
            pred_boxes,
            true_boxes,
            iou_threshold=config.MAP_IOU_THRESH,
            box_format="midpoint",
            num_classes=config.NUM_CLASSES,
        )
        writer.add_scalars('accuracy', {'class_acc_train': class_acc_train,
                                        'noobj_acc_train': noobj_acc_train,
                                        'obj_acc_train': obj_acc_train,
                                        'class_acc_test': class_acc_test,
                                        'noobj_acc_test': noobj_acc_test,
                                        'obj_acc_test': obj_acc_test,
                                        'map': mapval,
                                        }, epoch
                           )
        model.train()


if __name__ == "__main__":
    writer = SummaryWriter(config.WRITER)
    main(writer)
