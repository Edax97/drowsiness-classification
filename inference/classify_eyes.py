import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers import ClassificationResult
from mediapipe.tasks.python.vision import ImageClassifier

from classifier_run import get_result

def create_eye_classifier(model_path: str, min_score=0.55, result_cb=None) -> ImageClassifier:
    options = vision.ImageClassifierOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.LIVE_STREAM,
            max_results=1,
            score_threshold=min_score,
            result_callback=result_cb
        )
    classifier = vision.ImageClassifier.create_from_options(options)
    return classifier

def classify_eyes(classifier: ImageClassifier, left_img: np.ndarray, right_img: np.ndarray, ms: int):
    left_img, right_img = np.ascontiguousarray(left_img), np.ascontiguousarray(right_img)
    left_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=left_img)
    right_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=right_img)
    l_ms = ms if ms % 2 == 1 else ms + 1
    r_ms = ms + 1 if ms % 2 == 1 else ms + 2
    classifier.classify_async(left_mp, l_ms)
    classifier.classify_async(right_mp, r_ms)

AWAKE_CLASS="awake"
DROWSY_CLASS="sleepy"
NO_CLASS=""
MIN_ONEEYE_AWAKE_SCORE = 0.6
MIN_ONEEYE_DROWSY_SCORE = 0.6
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
    return decide_category(left_category, left_score, right_category, right_score)
