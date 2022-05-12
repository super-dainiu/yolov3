import cv2
import matplotlib
import numpy as np
import config


# Settings
matplotlib.rc('font', **{'size': 11})


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


def plot_one_box(x, im, color=(128, 128, 128), label=None, line_thickness=3):
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, tl, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def plot_boxes(image, boxes, box_format="midpoint", targets=config.PASCAL_CLASSES):
    cmap = Colors()
    class_labels = config.PASCAL_CLASSES
    colors = [cmap(i) for i in range(len(class_labels))]
    im = np.ascontiguousarray(image)
    height, width, _ = im.shape

    for box in boxes:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"

        if class_labels[int(box[0])] not in targets:
            continue
        class_pred = box[0]
        class_prob = box[1]
        box = box[2:]
        if box_format == "midpoint":
            upper_left_x = box[0] - box[2] / 2
            upper_left_y = box[1] - box[3] / 2
            lower_right_x = box[0] + box[2] / 2
            lower_right_y = box[1] + box[3] / 2
        else:
            upper_left_x, upper_left_y, lower_right_x, lower_right_y = box
        plot_one_box((upper_left_x * width, upper_left_y * height, lower_right_x * width, lower_right_y * height),
                     im, color=colors[int(class_pred)],
                     label=class_labels[int(class_pred)] + f"{class_prob * 100:.3f}%")
    return im