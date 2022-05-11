import numpy as np
import pandas as pd
import os


def box_iou_calc(box1, box2):
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    lt = np.maximum(box1[:, None, :2], box2[:, :2])
    rb = np.minimum(box1[:, None, 2:], box2[:, 2:])

    inter = np.prod(np.clip(rb-lt, a_min=0, a_max=None), 2)
    return inter / (area1[:, None] + area2 - inter)


class ConfusionMatrix:
    def __init__(self, num_classes: int, confidence=0.5, iou_threshold=0.3):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.confidence = confidence
        self.iou_threshold = iou_threshold

    def process_batch(self, detections, labels: np.ndarray):

        """
        :param detections: (Array[N, 6]), class, conf, x_min, y_min, x_max, y_max
        :param labels: (Array[M, 5]), class, x_min, y_min, x_max, y_max
        :return:
        """

        gt_classes = labels[:, 0].astype(np.int16)
        try:
            detections = detections[detections[:, 2] > self.confidence]
        except IndexError or TypeError:
            # for i, label in enumerate(labels):
            #     gt_class = gt_classes[i]
            #     self.matrix[self.num_classes, gt_class] += 1
            return

        detection_classes = detections[:, 0].astype(np.int16)
        all_iou = box_iou_calc(labels[:, 1:], detections[:, 2:])
        want_idx = np.where(all_iou > self.iou_threshold)
        all_matches = [[want_idx[0][i], want_idx[1][i], all_iou[want_idx[0][i], want_idx[1][i]]]
                       for i in range(want_idx[0].shape[0])]
        all_matches = np.array(all_matches)
        if all_matches.shape[0] > 0:  # if there is match
            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]
            all_matches = all_matches[np.unique(all_matches[:, 1], return_index=True)[1]]
            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]
            all_matches = all_matches[np.unique(all_matches[:, 0], return_index=True)[1]]
        for i, label in enumerate(labels):
            gt_class = gt_classes[i]
            if all_matches.shape[0] > 0 and all_matches[all_matches[:, 0] == i].shape[0] == 1:
                detection_class = detection_classes[int(all_matches[all_matches[:, 0] == i, 1][0])]
                self.matrix[detection_class, gt_class] += 1
            # else:
            #     self.matrix[self.num_classes, gt_class] += 1
        # for i, detection in enumerate(detections):
        #     if not all_matches.shape[0] or (all_matches.shape[0] and all_matches[all_matches[:, 1] == i].shape[0] == 0):
        #         detection_class = detection_classes[i]
        #         # self.matrix[detection_class, self.num_classes] += 1

    def return_matrix(self):
        return self.matrix

    def print_matrix(self):
        for i in range(self.num_classes + 1):
            print(' '.join(map(str, self.matrix[i])))


if __name__ == '__main__':
    class_dict = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5, 'car': 6, 'cat': 7,
                  'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14,
                  'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19}
    gt_path = './resnet50_predict_new/best_model_predict/ground-truth'
    det_path = './resnet50_predict_new/best_model_predict/detection-results'
    gt_files = os.listdir(gt_path)
    det_files = os.listdir(det_path)
    confusion_matrix = np.zeros((20, 20))
    for i in range(4952):
        gt_file_name = gt_files[i]
        det_file_name = det_files[i]
        gt_file = os.path.join(gt_path, gt_file_name)
        det_file = os.path.join(det_path, det_file_name)
        gt_matrix = []
        det_matrix = []
        for line in open(gt_file):
            line_info = line.split()
            if line_info[-1] == 'difficult':
                line_info.pop()
            line_info[0] = class_dict[line_info[0]]
            gt_line = [float(s) for s in line_info]
            # gt_line[0] = int(gt_line[0])
            gt_matrix.append(gt_line)
        for line in open(det_file):
            line_info = line.split()
            line_info[0] = class_dict[line_info[0]]
            det_line = [float(s) for s in line_info]
            # det_line[0] = int(det_line[0])
            det_matrix.append(det_line)
        gt_matrix = np.array(gt_matrix)
        det_matrix = np.array(det_matrix)
        conf_mat = ConfusionMatrix(num_classes=20, confidence=0.5, iou_threshold=0.3)
        conf_mat.process_batch(det_matrix, gt_matrix)
        confusion_matrix += conf_mat.return_matrix()
    # confusion_matrix = confusion_matrix[1:, 1:]
    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)
    TN = confusion_matrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    # print(TP)
    # print(TN)
    # print(FP)
    # print(FN)
    ACC = TP / (TP + FN)
    IOU = TP / (FN + FP + TP)
    print(np.mean(ACC))
    print(np.mean(IOU))
