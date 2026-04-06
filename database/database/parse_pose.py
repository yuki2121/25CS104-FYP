from pathlib import Path
from database.db import get_image_id, insert_image, insert_pose, insert_pose_vector
import json
import cv2

from shared.keypoints_manipulation import pose_2d_to_vector

import os
from dotenv import load_dotenv


# TODO: update to insert to 3d??

def is_blurry(image_path, bbox, threshold=150):
    img = cv2.imread(image_path)
    if img is None: return True
    
    x1, y1, x2, y2 = map(int, bbox)
    crop = img[y1:y2, x1:x2]
    
    if crop.size == 0: return True
    
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    return variance < threshold 

def parse_image_n_insert(path: str):
    coco18dir = Path(path) 
    for json_file in coco18dir.glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)
        image_path = data["image_path"]
        insert_image(image_path)

def parse_pose_n_insert(path: str):
    coco18dir = Path(path)
    for json_file in coco18dir.glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)
        try: 
            insert_data = {}
            image_id = get_image_id(data["image_path"])
            if image_id is None:
                print(f"Image not found for path: {data['image_path']}")
                continue
            insert_data["image_id"] = image_id
            insert_data["pose_json"] = json.dumps(data)
            bbox = data["bbox"][0]
            insert_data["bbox_top_x"] = bbox[0]
            insert_data["bbox_top_y"] = bbox[1]
            insert_data["bbox_bottom_x"] = bbox[2]
            insert_data["bbox_bottom_y"] = bbox[3]

            if is_blurry(data["image_path"], bbox):
                print(f"Skipping {data['image_path']} due to low clarity")
                continue

            scores = data["scores"]
            avg_confidence = sum(scores) / len(scores)

            if avg_confidence < 0.6: 
                print(f"Skipping: low average confidence ({avg_confidence:.2f})")
                continue

            bbox = data["bbox"][0]

            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            aspect_ratio = height / width

            # skip if too thin or flat
            if aspect_ratio > 4.0 or aspect_ratio < 0.2:
                print("Skipping: abnormal aspect ratio")
                continue


            insert_data["person_num"] = data["person_id"]
            insert_data["norm_joint"]=data["norm_type"]


            insert_pose(insert_data)
        except Exception as e:
            print(f"Error inserting pose for file {json_file}: {e}")

def parse_vector_n_insert(path: str):
    coco18dir = Path(path)
    for json_file in coco18dir.glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)
        try:
            insert_data = {}
            image_id = get_image_id(data["image_path"])
            if image_id is None:
                print(f"Image not found for path: {data['image_path']}")
                continue
            keypoints = data["norm_keypoints"]
            scores = data["scores"]
            pose_vector = pose_2d_to_vector(keypoints, scores)
            insert_data["image_id"] = image_id
            insert_data["pose_vector"] = pose_vector
            insert_data["person_num"] = data["person_id"]

            bbox = data["bbox"][0]

            if is_blurry(data["image_path"], bbox):
                print(f"Skipping {data['image_path']} due to low clarity")
                continue
            scores = data["scores"]
            avg_confidence = sum(scores) / len(scores)

            if avg_confidence < 0.6: 
                print(f"Skipping: low average confidence ({avg_confidence:.2f})")
                continue

            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            aspect_ratio = height / width

            # skip if too thin or flat
            if aspect_ratio > 4.0 or aspect_ratio < 0.2:
                print("Skipping: abnormal aspect ratio")
                continue


            insert_pose_vector(insert_data)
        except Exception as e:
            print(f"Error inserting pose vector for file {json_file}: {e}")


if __name__ == "__main__":
    load_dotenv()
    POSE_PATH = os.getenv("POSE_2D_JSON_PATH")
    parse_image_n_insert(POSE_PATH)
    parse_pose_n_insert(POSE_PATH)
    parse_vector_n_insert(POSE_PATH)