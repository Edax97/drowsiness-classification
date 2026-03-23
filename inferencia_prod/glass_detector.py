import cv2
import numpy as np
from PIL import Image
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates

from inferencia_prod.status_getter import StatusGetter

denormalize_coordinates = _normalized_to_pixel_coordinates

class GlassDetector:
    def __init__(self, time_threshold: float):
        self.time_threshold = time_threshold
        self.result = False
        self.last_result = False
        self.status = False
        self.status_getter = StatusGetter(time_threshold)

    def get_glasses(self, frame, landmarks, frame_width, frame_height):
        # GLASSES SETTINGS
        glass_id = [55, 285, 196, 419]
        coords_points = []

        for i in glass_id:
            lm = landmarks[i]
            coord = denormalize_coordinates(lm.x, lm.y, frame_width, frame_height)
            coords_points.append(coord)

        x_values = [coord[0] for coord in coords_points]
        y_values = [coord[1] for coord in coords_points]

        # Find the maximum and minimum values for x and y
        max_x = max(x_values)
        min_x = min(x_values)
        max_y = max(y_values)
        min_y = min(y_values)

        frame = Image.fromarray(frame.astype('uint8'), 'RGB')
        cropped = frame.crop((min_x, min_y, max_x, max_y))

        cropped_gray = cv2.cvtColor(np.array(cropped), cv2.COLOR_RGB2GRAY)
        region_variance = cv2.Laplacian(cropped_gray, cv2.CV_64F).var()

        #print(f"Variance: {region_variance:.3f}")

        self.last_result = self.result
        if region_variance > 45:
            self.result = True
        else:
            self.result = False
        return self.result, (min_x, min_y, max_x, max_y)

    def get_status(self):
        detection = self.result or self.last_result
        return self.status_getter.get_status(detection)