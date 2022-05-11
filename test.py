import config
from detect import *
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

RTSP_GSTREAMER_TEMPLATE = 'rtspsrc location={} ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv' \
                          ' ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink'
FILE_GSTREAMER_TEMPLATE = 'filesrc location={}  ! qtdemux ! h264parse ! nvv4l2decoder ! nvvidconv ' \
                          ' ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink'


if __name__ == "__main__":
    url = "samples/ronaldo.gif"
    cap = cv2.VideoCapture(url)

    # detector init
    detector = Detector(weights='pretrained/stage_2.pth.tar', conf_thres=0.7, iou_thres=0.3, max_det=1000,
                        opt_device=config.DEVICE, return_img=True, view_time=False
                        )

    # video config
    fps = max(cap.get(cv2.CAP_PROP_FPS) % 100, 0) or 25.0  # 25 FPS fallback

    sample_rate = round(0.5*fps)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fno = 0

    while True:
        cap.grab()

        image = cap.retrieve()[1]

        if image is None:
            break

        img0 = detector.detection_image(image)

        cv2.imshow(f'video', img0)

        if cv2.waitKey(1) == ord('q'):
            break

        fno = fno + 1

    cv2.destroyAllWindows()
