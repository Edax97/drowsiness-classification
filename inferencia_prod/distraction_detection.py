import cv2
import numpy as np
from PIL import Image
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

from inferencia_prod.status_getter import StatusGetter

class DistractionDetector:
    def __init__(self, ranges, time_threshold: float):
        self.head_up_range = ranges[0]
        self.head_down_range = ranges[1]
        self.head_left_range = ranges[2]
        self.head_right_range = ranges[3]

        self.head_result = "ENFRENTE"
        self.last_result = "ENFRENTE"
        self.status = False

        self.status_getter = StatusGetter(time_threshold)

    def get_head(self, _landmarks: list[NormalizedLandmark], img_h, img_w):
        face_2d = []
        face_3d = []
        for idx, lm in enumerate(_landmarks):
            if (
                    idx == 33
                    or idx == 263
                    or idx == 1
                    or idx == 61
                    or idx == 291
                    or idx == 199
            ):
                if idx == 1:
                    nose_2d = (lm.x * img_w, lm.y * img_h)
                    nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                x, y = int(lm.x * img_w), int(lm.y * img_h)
                face_2d.append([x, y])  # Get the 2D Coordinates
                face_3d.append([x, y, lm.z])  # Get the 3D Coordinates

        # Convert it to the NumPy array
        face_2d = np.array(face_2d, dtype=np.float64)
        # Convert it to the NumPy array
        face_3d = np.array(face_3d, dtype=np.float64)

        focal_length = 1 * img_w  # The camera matrix

        cam_matrix = np.array(
            [
                [focal_length, 0, img_h / 2],
                [0, focal_length, img_w / 2],
                [0, 0, 1],
            ]
        )

        # The distortion parameters
        dist_matrix = np.zeros((4, 1), dtype=np.float64)
        success, rot_vec, trans_vec = cv2.solvePnP(
            face_3d, face_2d, cam_matrix, dist_matrix)  # Solve PnP
        rmat, jac = cv2.Rodrigues(rot_vec)  # Get rotational matrix
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)  # Get angles

        # Get the y rotation degree
        x = angles[0] * 360
        y = angles[1] * 360
        z = angles[2] * 360

        if y < self.head_left_range * -1:
            head_position = "IZQ"

        elif y > self.head_right_range:
            head_position = "DER"

        elif x < self.head_down_range * -1:
            head_position = "ABAJO"

        elif x > self.head_up_range:
            head_position = "ARRIBA"

        else:
            head_position = "ENFRENTE"

        self.last_result = self.head_result
        self.head_result = head_position
        return head_position

    def get_status(self):
        if self.head_result == "ENFRENTE" or self.last_result == "ENFRENTE":
            detection = False
        else:
            detection = True
        return self.status_getter.get_status(detection)