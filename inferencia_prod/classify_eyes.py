import time

import numpy as np
import mediapipe as mp
from mediapipe import Image
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers import ClassificationResult

from inferencia_prod.status_getter import StatusGetter

AWAKE_CLASS="awake"
DROWSY_CLASS="sleepy"
NO_CLASS=""
MIN_ONEEYE_AWAKE_SCORE = 0.6
MIN_ONEEYE_DROWSY_SCORE = 0.4
def decide_category(_left_category: str, _left_score: float, _right_category: str, _right_score: float) -> str:
    if _left_category == _right_category:
        return _left_category

    if not (_left_category == AWAKE_CLASS or _right_category == AWAKE_CLASS):
        drosy_eye_score = _left_score if _left_category == DROWSY_CLASS else _right_score
        if drosy_eye_score > MIN_ONEEYE_DROWSY_SCORE:
            return DROWSY_CLASS
        return NO_CLASS

    awake_eye_score = _left_score if _left_category == AWAKE_CLASS else _right_score
    if awake_eye_score > MIN_ONEEYE_AWAKE_SCORE:
        return AWAKE_CLASS

    return NO_CLASS

def get_status(l_result: ClassificationResult, r_result: ClassificationResult) -> str:
    left_category, left_score = get_result(l_result)
    right_category, right_score = get_result(r_result)
    print("Left", left_category, left_score)
    print("Right", right_category, right_score)
    print("---")
    return decide_category(left_category, left_score, right_category, right_score)

def get_result(result: ClassificationResult) -> tuple[str, float]:
    category = ""
    score = 0
    if len(result.classifications[0].categories) > 0:
        top_category = result.classifications[0].categories[0]
        category = top_category.category_name
        score = top_category.score
    return category, score

MINIMUM_EYE_CLASSIFICATION_TIME=0.3
class EyeClassifier:
    left_result: ClassificationResult

    def __init__(self, model_path: str, min_score: float, callback, close_time: float):
        self.callback = callback
        self.class_result = NO_CLASS
        self.last_class_results = [NO_CLASS, NO_CLASS, NO_CLASS, NO_CLASS]

        self.eye_getter = StatusGetter(close_time)

        self.last_classification=time.time()

        options = vision.ImageClassifierOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.LIVE_STREAM,
            max_results=1,
            score_threshold=min_score,
            result_callback=self.eye_classifier_cb
        )
        self.classifier = vision.ImageClassifier.create_from_options(options)

    def eye_classifier_cb(self, result: ClassificationResult, _: Image, ms: int):
        if ms % 2 == 1:
            self.left_result = result
            return
        status = get_status(result, self.left_result)
        if self.callback is not None:
            self.callback(status)
        self.class_result = status

    def classify_eyes(self, left_img: np.ndarray, right_img: np.ndarray):
        if time.time() - self.last_classification < MINIMUM_EYE_CLASSIFICATION_TIME:
            self.last_classification = time.time()
            return
        ms = int(1000 * time.time())
        left_img, right_img = np.ascontiguousarray(left_img), np.ascontiguousarray(right_img)
        left_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=left_img)
        right_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=right_img)
        l_ms = ms if ms % 2 == 1 else ms + 1
        r_ms = ms + 1 if ms % 2 == 1 else ms + 2
        self.classifier.classify_async(left_mp, l_ms)
        self.classifier.classify_async(right_mp, r_ms)

    def get_result(self):
        return self.class_result

    def get_status(self):
        self.last_class_results[:-1] = self.last_class_results[1:]
        self.last_class_results[-1] = self.class_result

        if self.class_result == AWAKE_CLASS:
            return self.eye_getter.get_status(False)

        if self.class_result == DROWSY_CLASS:
            return self.eye_getter.get_status(True)

        num_drowsy=0
        for r in self.last_class_results:
            if r == DROWSY_CLASS:
                num_drowsy += 1

        if num_drowsy * 2 >= len(self.last_class_results):
            return self.eye_getter.get_status(True)

        return self.eye_getter.get_status(False)


