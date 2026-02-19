import json
import sys

import cv2
import os

MIN_SIZE = 0.15
HEIGHT = 320


def crop_objects(json_path, images_dir, output_dir, ids: list[str]):
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    images = {img['id']: img for img in coco_data['images']}

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in ids:
        if not os.path.exists(os.path.join(output_dir, f"{i}")):
            os.makedirs(os.path.join(output_dir, f"{i}"))

    for j, ann in enumerate(coco_data['annotations']):
        image_info = images.get(ann['image_id'])
        if not image_info:
            continue

        image_path = os.path.join(images_dir, image_info['file_name'])
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found.")
            continue

        img = cv2.imread(f"{image_path}")
        if img is None:
            print(f"Warning: Could not read image {image_path}.")
            continue

        x, y, w, h = ann['bbox']
        if h < img.shape[0] * MIN_SIZE:
            continue

        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = img[max(0, y1):min(img.shape[0], y2), max(0, x1):min(img.shape[1], x2)]
        if crop.size == 0:
            continue

        # Resize to height 320
        aspect_ratio = crop.shape[1] / crop.shape[0]
        new_width = int(HEIGHT * aspect_ratio)
        resized_crop = cv2.resize(crop, (new_width, HEIGHT), interpolation=cv2.INTER_LINEAR)
        resized_crop = cv2.cvtColor(resized_crop, cv2.COLOR_BGR2GRAY)

        # Save
        output_class = f"{ann['category_id']}"
        output_filename = f"img_{j}.jpg"
        output_path = os.path.join(output_dir, output_class, output_filename)
        cv2.imwrite(output_path, resized_crop)

    print(f"Finished cropping {len(coco_data['annotations'])} objects.")


if __name__ == "__main__":
    # train
    base_dir = sys.argv[1]
    split = sys.argv[2]
    categories = sys.argv[3:]
    _json_path = f"{base_dir}/{split}/_annotations.coco.json"
    _images_dir = f"{base_dir}/{split}"
    _output_dir = f"{base_dir}/data_{split}"
    crop_objects(_json_path, _images_dir, _output_dir, categories)


