import json
import os
import sys

import cv2

MIN_SIZE = 0.15
HEIGHT = 320

def move_frames(json_path, images_dir, output_dir, class_list: list[str]):
    with open(json_path, 'r') as f:
        annotations = json.load(f)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in class_list:
        if not os.path.exists(os.path.join(output_dir, i)):
            os.makedirs(os.path.join(output_dir, i))

    for j, img_file in enumerate(annotations.keys()):
        img_info = annotations[img_file]
        class_name = img_info["driver_state"]
        if not img_info or not class_name or class_name not in class_list:
            continue
        img_file = img_file.replace("./", "")
        img_path = f"{images_dir}/{img_file}"
        if not os.path.exists(img_path):
            print(f"image {img_path} not found.")
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(f"{output_dir}/{class_name}/img_{j}.jpg", img)

    print(f"Finished moving {len(annotations.keys())} objects.")

if __name__ == "__main__":
    split = sys.argv[1]
    categories = sys.argv[2:]
    _json_path = f"datasets/FL3D/classification_frames/annotations_{split}.json"
    _images_dir = "datasets/FL3D"
    _output_dir = f"datasets/FL3D/data_{split}"
    move_frames(_json_path, _images_dir, _output_dir, categories)

