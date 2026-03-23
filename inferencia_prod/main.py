import os
import subprocess

import cv2

from classify_eyes import EyeClassifier
from distraction_detection import DistractionDetector
from face_landmarks import create_landmarker
from fatiga_process import process_frame
from glass_detector import GlassDetector
from inferencia_prod.object_detector import ObjectDetector
from serial_send import SerialSender, ScannerListener
from dsm_status import DSMstatus
from camera import find_camera

Frame_width = 900
Frame_height = 540
LANDMARKER_PATH = "face_landmarker.task"
# ------
# Fatiga
EYE_CLASSIFIER_PATH = "eyes_model/en0_eye_b.tflite"
EYE_CLOSE_TIME = 3
EYE_SCORE = 0.5
# -----------
# Distraction
LR_HEAD_RANGE = 17
UP_HEAD_RANGE = 25
DOWN_HEAD_RANGE = 5
TIEMPO_DISTRACCION=5
# -------
# Glasses
TIEMPO_GAFAS=3
# ------------
# Object detection
OBJECT_MODEL_FILENAME="object_model/model_28_08.tflite"
ALLOW_LIST=[
    "cigarro",
    "cinturon",
    "celular"
]
THRESHOLD_TELEFONO=0.25
THRESHOLD_CIGARRO=0.35
THRESHOLD_CINTURON=0.45
# -----------
# DSM
SEND_INTERVAL=120
MINIMUM_SEND_INTERVAL=0.5
# ----------
# Serial com
SERIAL_PORT="/dev/ttyAMA4"
BAUDRATE=115200
if os.environ.get("USER") != "pi":
    socat_process = subprocess.run("pgrep socat", shell=True, capture_output=True, text=True)
    if len(socat_process.stdout.strip()) == 0:
        subprocess.run("socat -d -d pty,raw,echo=0,link=/tmp/ttyV0 pty,raw,echo=0,link=/tmp/ttyV1 &", shell=True)
    SERIAL_PORT="/tmp/ttyV0"

if __name__ == "__main__":
    landmarker = create_landmarker(LANDMARKER_PATH)
    distraction_detector = DistractionDetector((UP_HEAD_RANGE, DOWN_HEAD_RANGE, LR_HEAD_RANGE, LR_HEAD_RANGE), TIEMPO_DISTRACCION)
    glass_detector = GlassDetector(TIEMPO_GAFAS)

    eye_classifier = EyeClassifier(EYE_CLASSIFIER_PATH, EYE_SCORE, None, EYE_CLOSE_TIME)
    
    object_detector = ObjectDetector(OBJECT_MODEL_FILENAME, ALLOW_LIST, THRESHOLD_TELEFONO, THRESHOLD_CIGARRO, THRESHOLD_CINTURON)

    serial_sender = SerialSender(SERIAL_PORT, BAUDRATE)
    scanner_listener = ScannerListener()

    dsm_status = DSMstatus(SEND_INTERVAL, MINIMUM_SEND_INTERVAL)

    cap = find_camera()
    # cap = cv2.VideoCapture("gafas_videos/vid6.mp4")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if os.environ.get("USER") != "pi":
            frame = cv2.resize(frame, (Frame_width, Frame_height))
        out = process_frame(
            frame,
            landmarker,
            glass_detector=glass_detector,
            distraction_detector=distraction_detector,
            eye_classifier=eye_classifier,
            object_detector=object_detector,
            dsm_status=dsm_status,
        )
        scanner_data = scanner_listener.get_data()
        if len(scanner_data) > 0:
            serial_sender.send_serial(scanner_data)

        dsm_data = dsm_status.get_status_string()
        if len(dsm_data) > 0:
            serial_sender.send_serial(dsm_data)

        cv2.imshow("Deteccion de fatiga", out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    scanner_listener.join()
