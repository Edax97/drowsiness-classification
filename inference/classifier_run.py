import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers.classification_result import ClassificationResult
from mediapipe.tasks.python.vision.image_classifier import ImageClassifier


MODELPATH="../classifier-exported/en0_D.tflite"
def create_clasifier(model_path: str=MODELPATH,min_score=0.55) -> ImageClassifier:
    options = vision.ImageClassifierOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        max_results=1,
        score_threshold=min_score,
    )
    classifier = vision.ImageClassifier.create_from_options(options)
    return classifier

def classify_image_async(classifier: ImageClassifier, image: np.ndarray, timestamp_ms: int):
    image_rgb = np.ascontiguousarray(image)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    classifier.classify_async(mp_image, timestamp_ms)

def classify_image(classifier: ImageClassifier, image: np.ndarray) -> tuple[str, float]:
    image_rgb = np.ascontiguousarray(image)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = classifier.classify(mp_image)
    return get_result(result)

def get_result(result: ClassificationResult) -> tuple[str, float]:
    category = ""
    score = 0
    if len(result.classifications[0].categories) > 0:
        top_category = result.classifications[0].categories[0]
        category = top_category.category_name
        score = top_category.score
    return category, score
