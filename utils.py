import config
import torch
import os
import pprint

from collections import Counter
from tqdm import tqdm

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def iou_width_height(boxes1, boxes2):
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
        boxes1[..., 1], boxes2[..., 1]
    )
    union = (
        boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
    )
    return intersection / union


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_suppression(bboxes, iou_threshold, threshold, box_format="midpoint", max_det=1000):

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes and len(bboxes_after_nms) < max_det:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]
        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def cells_to_bboxes(predictions, anchors, S, is_preds=True):
    BATCH_SIZE = predictions.shape[0]
    num_anchors = len(anchors)
    box_predictions = predictions[..., 1:5]
    if is_preds:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
        scores = torch.sigmoid(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)
    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]

    cell_indices = (
        torch.arange(S)
        .repeat(predictions.shape[0], 3, S, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )
    x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    w_h = 1 / S * box_predictions[..., 2:4]
    converted_bboxes = torch.cat((best_class, scores, x, y, w_h), dim=-1).reshape(BATCH_SIZE, num_anchors * S * S, 6)
    return converted_bboxes.tolist()


def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20
):

    # list storing all AP for respective classes
    average_precisions = {class_name: 0 for class_name in config.PASCAL_CLASSES}

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

            # print(best_iou)
        # print(TP, iou_threshold)
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions[config.PASCAL_CLASSES[c]] = torch.trapz(precisions, recalls).item() * 100

    print("Class mAP:")
    pprint.pprint(average_precisions)
    return sum(average_precisions.values()) / len(average_precisions)


def check_accuracy(out, labels, threshold,
                   tot_class_preds: torch.Tensor, correct_class: torch.Tensor,
                   tot_noobj: torch.Tensor, correct_noobj: torch.Tensor,
                   tot_obj: torch.Tensor, correct_obj: torch.Tensor
                   ):
    with torch.no_grad():
        for i in range(3):
            labels[i] = labels[i].to(config.DEVICE)
            obj = labels[i][..., 0] == 1
            noobj = labels[i][..., 0] == 0

            correct_class.add_(torch.sum(
                torch.argmax(out[i][..., 5:][obj], dim=-1) == labels[i][..., 5][obj]
            ))
            tot_class_preds.add_(torch.sum(obj))

            obj_preds = torch.sigmoid(out[i][..., 0]) > threshold
            correct_obj.add_(torch.sum(obj_preds[obj] == labels[i][..., 0][obj]))
            tot_obj.add_(torch.sum(obj))
            correct_noobj.add_(torch.sum(obj_preds[noobj] == labels[i][..., 0][noobj]))
            tot_noobj.add_(torch.sum(noobj))


def get_bboxes(out, labels, iou_thres, conf_thres, anchors, all_pred_boxes, all_true_boxes, batch_idx, box_format="midpoint", device="cuda"):
    with torch.no_grad():
        batch_size = out[0].shape[0]
        bboxes = [[] for _ in range(batch_size)]
        for i in range(3):
            S = out[i].shape[2]
            anchor = torch.tensor([*anchors[i]]).to(device) * S
            boxes_scale_i = cells_to_bboxes(
                out[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

        # we just want one bbox for each label, not one for each scale
        true_bboxes = cells_to_bboxes(
            labels[2], anchor, S=S, is_preds=False
        )
        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_thres,
                threshold=conf_thres,
                box_format=box_format,
            )
            for nms_box in nms_boxes:
                all_pred_boxes.append([batch_idx * batch_size + idx] + nms_box)

            for box in true_bboxes[idx]:
                if box[1] > conf_thres:
                    all_true_boxes.append([batch_idx * batch_size + idx] + box)


def get_evaluation_bboxes(loader, model, iou_threshold, anchors, threshold, box_format="midpoint", device="cuda",):
    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0
    all_pred_boxes = []
    all_true_boxes = []
    for batch_idx, (x, labels) in enumerate(tqdm(loader)):
        x = x.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        bboxes = [[] for _ in range(batch_size)]
        for i in range(3):
            S = predictions[i].shape[2]
            anchor = torch.tensor([*anchors[i]]).to(device) * S
            boxes_scale_i = cells_to_bboxes(
                predictions[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

        # we just want one bbox for each label, not one for each scale
        true_bboxes = cells_to_bboxes(
            labels[2], anchor, S=S, is_preds=False
        )

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes


def train_iter(loader, model, optimizer, loss_fn, scaler, scheduler, scaled_anchors, writer, epoch):
    loop = tqdm(loader, leave=True)
    losses = []
    metrics = [torch.tensor(0).to(config.DEVICE) for _ in range(6)]

    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = [_.to(config.DEVICE) for _ in y]

        with torch.cuda.amp.autocast():
            out = model(x)
            loss = loss_fn(out, y, scaled_anchors)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.detach().item())
        check_accuracy(out, y, config.CONF_THRESHOLD, *metrics)
        loop.set_postfix(loss=sum(losses) / len(losses), lr=scheduler.get_last_lr())

    mean_loss = sum(losses) / len(losses)
    metrics = [int(_) for _ in metrics]
    tot_class_preds, correct_class, tot_noobj, correct_noobj, tot_obj, correct_obj = metrics

    writer.add_scalars('loss', {'train_loss': mean_loss}, epoch)
    writer.add_scalar('learning rate', scheduler.get_last_lr()[0], epoch)
    print(f"Class accuracy is: {(correct_class / (tot_class_preds + 1e-16)) * 100:2f}%")
    print(f"No obj accuracy is: {(correct_noobj / (tot_noobj + 1e-16)) * 100:2f}%")
    print(f"Obj accuracy is: {(correct_obj / (tot_obj + 1e-16)) * 100:2f}%")
    writer.add_scalars('accuracy', {'class_acc_train': (correct_class/(tot_class_preds+1e-16))*100,
                                    'noobj_acc_train': (correct_noobj/(tot_noobj+1e-16))*100,
                                    'obj_acc_train': (correct_obj/(tot_obj+1e-16))*100,
                                    }, epoch)

    if (epoch + 1) % 10 == 0:
        all_pred_boxes, all_true_boxes = get_evaluation_bboxes(loader, model, config.NMS_IOU_THRESH, config.ANCHORS, config.CONF_THRESHOLD)
        mapval = mean_average_precision(
            all_pred_boxes,
            all_true_boxes,
            iou_threshold=config.MAP_IOU_THRESH,
            box_format="midpoint",
            num_classes=config.NUM_CLASSES,
        )
        print(f"Train mAP: {mapval}")
        writer.add_scalars('mAP', {'mAP_train': mapval}, epoch)


def test_iter(loader, model, loss_fn, scaled_anchors, writer, epoch):
    with torch.no_grad():
        loop = tqdm(loader, leave=True)
        losses = []
        metrics = [torch.tensor(0).to(config.DEVICE) for _ in range(6)]

        for batch_idx, (x, y) in enumerate(loop):
            x = x.to(config.DEVICE)
            y = [_.to(config.DEVICE) for _ in y]

            out = model(x)
            loss = loss_fn(out, y, scaled_anchors)
            losses.append(loss.detach().item())
            check_accuracy(out, y, config.CONF_THRESHOLD, *metrics)
            loop.set_postfix(loss=sum(losses) / len(losses))

        mean_loss = sum(losses) / len(losses)
        metrics = [int(_) for _ in metrics]
        tot_class_preds, correct_class, tot_noobj, correct_noobj, tot_obj, correct_obj = metrics

    writer.add_scalars('loss', {'test_loss': mean_loss}, epoch)
    print(f"Class accuracy is: {(correct_class / (tot_class_preds + 1e-16)) * 100:2f}%")
    print(f"No obj accuracy is: {(correct_noobj / (tot_noobj + 1e-16)) * 100:2f}%")
    print(f"Obj accuracy is: {(correct_obj / (tot_obj + 1e-16)) * 100:2f}%")
    writer.add_scalars('accuracy', {'class_acc_test': (correct_class / (tot_class_preds + 1e-16)) * 100,
                                    'noobj_acc_test': (correct_noobj / (tot_noobj + 1e-16)) * 100,
                                    'obj_acc_test': (correct_obj / (tot_obj + 1e-16)) * 100,
                                    }, epoch)
    if (epoch + 1) % 10 == 0:
        all_pred_boxes, all_true_boxes = get_evaluation_bboxes(loader, model, config.NMS_IOU_THRESH, config.ANCHORS,
                                                               config.CONF_THRESHOLD)
        mapval = mean_average_precision(
            all_pred_boxes,
            all_true_boxes,
            iou_threshold=config.MAP_IOU_THRESH,
            box_format="midpoint",
            num_classes=config.NUM_CLASSES,
        )
        print(f"Test mAP: {mapval}")
        writer.add_scalars('mAP', {'mAP_test': mapval}, epoch)
