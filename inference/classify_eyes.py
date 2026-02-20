import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.vision import ImageClassifier

from classifier_run import get_result


def classify_eyes(classifier: ImageClassifier, left_img: np.ndarray, right_img: np.ndarray):
    left_img, right_img = np.ascontiguousarray(left_img), np.ascontiguousarray(right_img)
    left_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=left_img)
    right_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=right_img)

    left_result = classifier.classify(left_mp)
    right_result = classifier.classify(right_mp)

    left_category, left_score = get_result(left_result)
    right_category, right_score = get_result(right_result)
    print(f"---\nleft_category: {left_category}, left_score: {left_score}")
    print(f"right_category: {right_category}, right_score: {right_score}")
    category = decide_category(left_category, left_score, right_category, right_score)
    print(f"category: {category}")
    return category

AWAKE_CLASS="awake"
DROWSY_CLASS="sleepy"
NO_CLASS=""
MIN_ONEEYE_AWAKE_SCORE = 0.75
MIN_ONEEYE_DROWSY_SCORE = 0.55
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

    if _left_category == NO_CLASS or _right_category == NO_CLASS:
        return NO_CLASS

    sleep_eye_score = _left_score if _left_category == DROWSY_CLASS else _right_score
    if sleep_eye_score > awake_eye_score:
        return DROWSY_CLASS
    return NO_CLASS



