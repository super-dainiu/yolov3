import config
from detect import (
    Detector,
    imread,
)
import os
import cv2
import argparse
import random
import imageio

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

RTSP_GSTREAMER_TEMPLATE = 'rtspsrc location={} ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv' \
                          ' ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink'
FILE_GSTREAMER_TEMPLATE = 'filesrc location={}  ! qtdemux ! h264parse ! nvv4l2decoder ! nvvidconv ' \
                          ' ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink'


def dir_path(string):
    if os.path.isdir(os.path.join(os.getcwd(), string)):
        return string
    else:
        raise NotADirectoryError(os.path.join(os.getcwd(), string))


def file_path(string):
    if os.path.isfile(os.path.join(os.getcwd(), string)):
        return string
    else:
        raise NotADirectoryError(os.path.join(os.getcwd(), string))


def img_detect(detector, img_dir, save_dir, save=False):
    img0 = imread(img_dir)
    img0 = detector.detection_image(img0)
    cv2.imshow('image', img0)
    if save:
        save_dir = save_dir + "/" + str(random.randint(0, 114514)) + ".jpg"
        print(f'Saving to {save_dir}')
        cv2.imwrite(save_dir, img0)
    cv2.waitKey(0)


def video_detect(detector, url, save_dir, save=False):
    cap = cv2.VideoCapture(url)

    # video config
    fps = max(cap.get(cv2.CAP_PROP_FPS) % 100, 0) or 25.0  # 25 FPS fallback

    sample_rate = round(0.5 * fps)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fno = 0
    frames = []

    while True:
        cap.grab()

        image = cap.retrieve()[1]

        if image is None:
            break

        img0 = detector.detection_image(image)

        if save:
            frames.append(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB))

        cv2.imshow(f'video', img0)

        if cv2.waitKey(1) == ord('q'):
            break

        fno = fno + 1

    cv2.destroyAllWindows()
    if save:
        save_dir = save_dir + "/" + str(random.randint(0, 114514)) + ".gif"
        print(f'Saving to {save_dir}')
        imageio.mimsave(save_dir, frames, fps=fps)


def main(args):
    # detector init
    detector = Detector(weights=args.weights,
                        conf_thres=args.conf_thres,
                        iou_thres=args.iou_thres,
                        max_det=args.max_det,
                        opt_device=config.DEVICE,
                        target=args.target,
                        view_time=False
                        )

    if args.camera:
        video_detect(detector, 0, args.save_dir)
        return 0

    for sample_dir in args.samples:
        if sample_dir.endswith('.png') or sample_dir.endswith('.jpg'):
            img_detect(detector, sample_dir, args.save_dir, args.save)

        elif sample_dir.endswith('.gif') or sample_dir.endswith('.mp4'):
            video_detect(detector, sample_dir, args.save_dir, args.save)

        else:
            raise NotImplementedError("Unexpected file type")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Use YOLOv3')
    parser.add_argument('--weights', type=file_path, required=True, help='Weight directory')
    parser.add_argument('--save', type=bool, default=False, help='Save results')
    parser.add_argument('--save_dir', type=dir_path, default='outputs', help='Save directory')
    parser.add_argument('--conf_thres', type=float, default=0.75, help='Confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.3, help='IOU threshold')
    parser.add_argument('--max_det', type=int, default=100, help='Maximum detection per frame')
    parser.add_argument('--target', type=str, nargs='+', default=("*", ), help='Targets (i.e. person), * for all classes')
    parser.add_argument('--camera', type=bool, default=False, help='Use your camera')
    parser.add_argument('--samples', type=file_path, nargs='+', help='Sample images (ends with .jpg, .png, .gif, .mp4)')
    args = parser.parse_args()
    main(args)
