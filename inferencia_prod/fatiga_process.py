
import cv2

from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker

import utils
from classify_eyes import EyeClassifier, DROWSY_CLASS
from distraction_detection import DistractionDetector
from face_landmarks import get_face, draw_lm, get_eyes
from glass_detector import GlassDetector
from dsm_status import DSMstatus
from object_detector import ObjectDetector

distraction_list = ["IZQ", "DER", "ABAJO", "ARRIBA"]
def process_frame(
        frame: cv2.Mat,
        _landmarker: FaceLandmarker,
        glass_detector: GlassDetector,
        distraction_detector: DistractionDetector,
        eye_classifier: EyeClassifier,
        object_detector: ObjectDetector,
        dsm_status: DSMstatus
    ) -> cv2.Mat:
    frame_height, frame_width, _ = frame.shape

    f_landmarks = get_face(_landmarker, frame)
    if f_landmarks is None:
        dsm_status.set_status(driver=False)
        return frame

    # Gafas
    result_gafas, _ = glass_detector.get_glasses(frame, f_landmarks, frame_width, frame_height)
    status_gafas = glass_detector.get_status()
    # cv2.rectangle(frame, (gx, gy), (gx1, gy1), (0, 230, 255), 1)

    # Distraccion
    head_position = distraction_detector.get_head(f_landmarks, frame_height, frame_height)
    status_distraccion = distraction_detector.get_status()

    # Fatiga
    (lx, ly, rx, ry), (lx1, ly1, rx1, ry1) = get_eyes(frame_width, frame_height, f_landmarks)
    left_roi = frame[ly:ly1, lx:lx1].copy()
    right_roi = frame[ry:ry1, rx:rx1].copy()
    eye_classifier.classify_eyes(left_roi, right_roi)
    result_fatiga = eye_classifier.get_result()
    status_fatiga = eye_classifier.get_status()
    cv2.rectangle(frame, (lx, ly), (lx1, ly1), (0, 0, 255), 1)
    cv2.rectangle(frame, (rx, ry), (rx1, ry1), (0, 0, 255), 1)

    # Objetos
    result_cigarro, result_cinturon, result_telefono = object_detector.detect_objects(frame)
    status_cigarro, status_cinturon, status_telefono = object_detector.get_status()

    # Display
    utils.display(frame, f"- STATUS_GAFAS: {status_gafas}", 1)
    utils.display(frame, f"- DISTRACCION: {head_position}", 2)
    utils.display(frame, f"  STATUS_DISTRACCION: {status_distraccion}", 3)
    utils.display(frame, f"- FATIGA: {result_fatiga}", 4)
    utils.display(frame, f"  STATUS_FATIGA: {status_fatiga}", 5)
    utils.display(frame, f"- TELEFONO: {result_telefono}", 6)
    utils.display(frame, f"  STATUS_TELEFONO: {status_telefono}", 7)

    frame = draw_lm(frame, f_landmarks)

    dsm_status.set_status(
        driver=True,
        telefono=False,
        cigarro=False,
        cinturon=True,
        gafas=status_gafas,
        distraccion=status_distraccion,
        fatiga=False)            

    return frame




