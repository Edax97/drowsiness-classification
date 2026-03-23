import time

import cv2
import numpy as np

from status_getter import StatusGetter
from tflite_support import task

MINIMUM_OBJECT_DETECTION_TIME=0.2
class ObjectDetector:
    def __init__(self,
                 model_filename: str,
                 allowlist: list[str],
                 threshold_cigarro=0.3,
                 threshold_cinturon=0.3,
                 threshold_telefono=0.3,
                 timeout_cigarro=3,
                 timeout_cinturon=30,
                 timeout_telefono=3,
                 ):
        options = task.vision.ObjectDetectorOptions(
            base_options=task.core.BaseOptions(file_name=model_filename),
            detection_options=task.processor.DetectionOptions(
                max_results=6,
                score_threshold=min(threshold_cigarro, threshold_cinturon, threshold_telefono),
                category_name_allowlist=allowlist,
            )
        )
        self.detector = task.vision.ObjectDetector.create_from_options(options)

        self.cigarro = False
        self.cinturon = True
        self.telefono = False

        self.getter_cigarro = StatusGetter(timeout_cigarro)
        self.getter_cinturon = StatusGetter(timeout_cinturon)
        self.getter_telefono = StatusGetter(timeout_telefono)

        self.threshold_cigarro = threshold_cigarro
        self.threshold_cinturon = threshold_cinturon
        self.threshold_telefono = threshold_telefono

        self.last_detection_time = time.time()
        self.name_cigarro, self.name_cinturon, self.name_telefono = allowlist

    def detect_objects(self, frame: np.ndarray):
        if time.time() - self.last_detection_time < MINIMUM_OBJECT_DETECTION_TIME:
            return self.cigarro, self.cinturon, self.telefono

        self.last_detection_time = time.time()
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = task.vision.TensorImage.create_from_array(img_rgb)

        results = self.detector.detect(img_tensor)
        cigarro, cinturon, telefono = False, False, False

        if len(results.detections) > 0:
            #print("---")
            for detection in results.detections:
                category_name = detection.categories[0].category_name
                score = detection.categories[0].score
                #print(f"Deteccion {category_name} con score {score}")
                if category_name == self.name_cigarro and score > self.threshold_cigarro:
                    cigarro = True
                elif category_name == self.name_cinturon and score > self.threshold_cinturon:
                    cinturon = True
                elif (category_name == self.name_telefono) and score > self.threshold_telefono:
                    telefono = True
                else:
                    continue

        self.cigarro, self.cinturon, self.telefono = cigarro, cinturon, telefono
        return self.cigarro, self.cinturon, self.telefono

    def get_status(self):
        status_cigarro = self.getter_cigarro.get_status(self.cigarro)
        status_not_cinturon = self.getter_cinturon.get_status(not self.cinturon)
        status_cinturon = not status_not_cinturon

        status_telefono = self.getter_telefono.get_status(self.telefono)

        return status_cigarro, status_cinturon, status_telefono

