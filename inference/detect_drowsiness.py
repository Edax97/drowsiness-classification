import os
import time

from mediapipe import Image
from mediapipe.tasks.python.components.containers import ClassificationResult

from cv_camera import view_cam, text_overlay
import cv2 as cv

from alert import alert_detection
from classify_eyes import get_status, classify_eyes, DROWSY_CLASS, AWAKE_CLASS, create_eye_classifier
from detection_header import Detection_Header
from face_det_mp import detect_face_mp

last_time = 0
if __name__ == "__main__":

    detector = Detection_Header(DROWSY_CLASS, AWAKE_CLASS)
    left_result: ClassificationResult
    def result_cb(result: ClassificationResult, _: Image, ms: int):
        global left_result
        if ms % 2 == 1:
            left_result = result
            return
        status = get_status(left_result, result)
        status = detector.set_status(status)
        alert_detection(status)

    classifier = create_eye_classifier("eyes_model/en0_eye_a.tflite", 0.55, result_cb=result_cb)

    def process_frame(frame: cv.Mat) -> cv.Mat:
        global last_time
        detected, (x, y, w, h), (l_x, l_y, l_x1, l_y1), (r_x, r_y, r_x1, r_y1) = detect_face_mp(frame)
        frame = cv.medianBlur(frame, 3)
        if not detected:
            text_overlay(frame, "No face", (60, 20))
            return frame

        time_ms = int(1000 * time.time())
        if time_ms - last_time > 300:
            last_time = time_ms
            left_roi = frame[l_y:l_y1, l_x:l_x1].copy()
            right_roi = frame[r_y:r_y1, r_x:r_x1].copy()
            classify_eyes(classifier, left_roi, right_roi, time_ms)
        drowsy_status = detector.get_status()
        output = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.rectangle(output, (l_x, l_y), (l_x1, l_y1), (255, 0, 0), 2)
        cv.rectangle(output, (r_x, r_y), (r_x1, r_y1), (255, 0, 0), 2)
        text_overlay(frame, f"Estado: {drowsy_status}", (20, 40))
        return output

    device_id = os.getenv("DEVICE") if os.getenv("DEVICE") else 0
    view_cam("Fatiga", device_id, process_frame)