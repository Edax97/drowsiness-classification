import json
import sys
from os.path import isdir

import cv2 as cv
import os

from inference.face_det_mp import detect_face_mp

MIN_SIZE = 0.15
HEIGHT = 320

def crop_eyes(images_dir, output_dir, ids: list[str]):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cropped = 0
    for i in ids:
        if not os.path.exists(os.path.join(output_dir, f"{i}")):
            os.makedirs(os.path.join(output_dir, f"{i}"))

        _class_dir_path = os.path.join(images_dir, f"{i}")
        for f in os.listdir(_class_dir_path):
            img_path = os.path.join(_class_dir_path, f)
            if isdir(img_path):
                continue

            img = cv.imread(f"{img_path}")
            if img is None:
                print(f"Warning: Could not read image {img_path}.")
                continue

            ok, face, (lx, ly, lx1, ly1), (rx, ry, rx1, ry1)= detect_face_mp(img)
            if not ok:
                continue
            left_crop = img[max(0, ly):min(img.shape[0], ly1), max(0, lx):min(img.shape[1], lx1)]
            right_crop = img[max(0, ry):min(img.shape[0], ry1), max(0, rx):min(img.shape[1], rx1)]

            l_eye_path = os.path.join(output_dir, f"{i}", f"left_{f}")
            r_eye_path = os.path.join(output_dir, f"{i}", f"right_{f}")
            cv.imwrite(l_eye_path, left_crop)
            cv.imwrite(r_eye_path, right_crop)
            cropped += 2
    print(f"Finished cropping {cropped} images.")


if __name__ == "__main__":
    img_dir = "../classification/datasets/FL3D/data_val"
    save_dir = "FL3D-eyesbig"
    crop_eyes(img_dir, save_dir, ["1", "2"])
