import mediapipe as mp
import cv2 as cv
from mediapipe.tasks.python.components.containers.keypoint import NormalizedKeypoint
from mediapipe.tasks.python.vision import FaceDetectorOptions

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path='face-model/blaze_face_short_range.tflite'),
    min_detection_confidence=0.67,
    running_mode=VisionRunningMode.IMAGE)
face_detector = mp.tasks.vision.FaceDetector.create_from_options(options)

def detect_face_mp(frame: cv.Mat):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = face_detector.detect(mp_image)
    faces = result.detections
    if len(faces) == 0:
        return False, (0,0,0,0), (0,0,0,0), (0,0,0,0)

    face_main = faces[0].bounding_box
    x, y, w, h = face_main.origin_x, face_main.origin_y, face_main.width, face_main.height

    lm = faces[0].keypoints
    left_kp = lm[0]
    right_kp = lm[1]
    return (True, (x,y,w,h),
            get_eye_coordinates(mp_image.width, mp_image.height, int(w*0.3), left_kp),
            get_eye_coordinates(mp_image.width, mp_image.height, int(w*0.3), right_kp, is_left=False))

def get_eye_coordinates(img_width: int, img_height: int, window_size: int, kp: NormalizedKeypoint, is_left=True):
    center_x, center_y = int(kp.x*img_width), int(kp.y*img_height)
    if is_left:
        h_prev = int(window_size*0.6)
        h_after = int(window_size*0.4)
    else:
        h_prev = int(window_size*0.4)
        h_after = int(window_size*0.6)
    x = center_x - h_prev if center_x > h_prev else 0
    x1 = center_x + h_after if center_x + h_after < img_width else img_width
    w_top = int(window_size*0.6)
    w_bottom = int(window_size*0.4)
    y = center_y - w_top if center_y > w_top else 0
    y1 = center_y + w_bottom if center_y + w_bottom < img_height else img_height
    return x,y,x1,y1


