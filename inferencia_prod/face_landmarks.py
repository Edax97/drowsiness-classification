import cv2
from mediapipe.tasks.python.vision import FaceLandmarker
import mediapipe as mp

from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions import drawing_utils
from mediapipe.python.solutions.drawing_utils import DrawingSpec

from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

def create_landmarker(model_path: str) -> FaceLandmarker:
    mp_options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        min_face_detection_confidence=0.5,
    )
    _landmarker = FaceLandmarker.create_from_options(mp_options)
    return _landmarker

def get_face(landmarker: FaceLandmarker, frame: cv2.Mat) -> list[NormalizedLandmark] | None:
    input_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = landmarker.detect(input_img)
    if len(result.face_landmarks) == 0:
        return None
    lm = result.face_landmarks[0]

    limits = [151, 200]
    for i in limits:
        l = lm[i]
        if l.x < 0 or l.y < 0:
            return None
    if len(lm) < 478:
        return None
    return lm

def get_eyes(frame_width: int, frame_height: int, _landmarks: list[NormalizedLandmark]):
    # MediaPipe face landmark indices for eyes
    # Left eye: 33, 133, 160, 159, 158, 157, 173, 144, 145, 153, 154, 155
    # Right eye: 362, 263, 387, 386, 385, 384, 398, 373, 374, 380, 381, 382
    # Center landmarks: left eye = 468, right eye = 473

    # Get face width by measuring distance between face edges
    left_face = _landmarks[234]  # left face edge
    right_face = _landmarks[454]  # right face edge
    face_width = int(abs(right_face.x - left_face.x) * frame_width)

    # Calculate window size (30% of face width)
    window_size = int(face_width * 0.3)
    half_size = window_size // 2

    # Left eye center (approximate center using landmarks)
    left_eye_center = _landmarks[468]
    left_center_x = int(left_eye_center.x * frame_width)
    left_center_y = int(left_eye_center.y * frame_height)

    # Right eye center
    right_eye_center = _landmarks[473]
    right_center_x = int(right_eye_center.x * frame_width)
    right_center_y = int(right_eye_center.y * frame_height)

    # Calculate bounding boxes (square aspect ratio)
    left_x = max(0, left_center_x - half_size)
    left_y = max(0, left_center_y - half_size)
    left_x1 = min(frame_width, left_center_x + half_size)
    left_y1 = min(frame_height, left_center_y + half_size)

    right_x = max(0, right_center_x - half_size)
    right_y = max(0, right_center_y - half_size)
    right_x1 = min(frame_width, right_center_x + half_size)
    right_y1 = min(frame_height, right_center_y + half_size)

    return (left_x, left_y, right_x, right_y), (left_x1, left_y1, right_x1, right_y1)

def draw_lm(frame: cv2.Mat, _landmarks: list[NormalizedLandmark]):
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in _landmarks
    ])
    drawing_utils.draw_landmarks(
        image=frame,
        landmark_list=face_landmarks_proto,
        landmark_drawing_spec=DrawingSpec(thickness=1, circle_radius=1, color=(255, 255, 255))
    )
    return frame