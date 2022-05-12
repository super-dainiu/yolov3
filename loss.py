import torch
import torch.nn as nn

from utils import intersection_over_union


class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        # Constants signifying how much to pay for each respective part of the loss
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

    def forward(self, predictions, targets, anchors):
        loss = 0
        for prediction, target, anchor in zip(predictions, targets, anchors):
            # Check where obj and noobj (we ignore if target == -1)
            obj = target[..., 0] == 1  # in paper this is Iobj_i
            noobj = target[..., 0] == 0  # in paper this is Inoobj_i

            # ======================= #
            #   FOR NO OBJECT LOSS    #
            # ======================= #

            no_object_loss = self.bce(
                (prediction[..., 0:1][noobj]), (target[..., 0:1][noobj]),
            )

            # ==================== #
            #   FOR OBJECT LOSS    #
            # ==================== #

            anchor = anchor.reshape(1, 3, 1, 1, 2)
            box_preds = torch.cat([self.sigmoid(prediction[..., 1:3]), torch.exp(prediction[..., 3:5]) * anchor], dim=-1)
            ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()
            object_loss = self.mse(self.sigmoid(prediction[..., 0:1][obj]), ious * target[..., 0:1][obj])

            # ======================== #
            #   FOR BOX COORDINATES    #
            # ======================== #

            prediction[..., 1:3] = self.sigmoid(prediction[..., 1:3])  # x,y coordinates
            target[..., 3:5] = torch.log(
                (1e-16 + target[..., 3:5] / anchor)
            )  # width, height coordinates
            box_loss = self.mse(prediction[..., 1:5][obj], target[..., 1:5][obj])

            # ================== #
            #   FOR CLASS LOSS   #
            # ================== #

            class_loss = self.entropy(
                (prediction[..., 5:][obj]), (target[..., 5][obj].long()),
            )

            loss += (
                self.lambda_box * box_loss
                + self.lambda_obj * object_loss
                + self.lambda_noobj * no_object_loss
                + self.lambda_class * class_loss
            )
        return loss
