# data preparation runs 2D estimation with HRNET, output json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from mmpose.apis import MMPoseInferencer
from mmdet.apis import DetInferencer
import inspect
import json
import os
from itertools import islice
import traceback

from shared.keypoints_manipulation import convert_2d_coco17_to_coco18

import os
from dotenv import load_dotenv

load_dotenv()

HRNET_CFG = "td-hm_hrnet-w32_8xb64-210e_coco-256x192"
DET_CFG   = "rtmdet_x_8xb32-300e_coco" 

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}



PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")).resolve() 

def image_path_to_json_path(image_path: Path, out_json_dir: Path, image_dir: Path):
    rel_path = image_path.relative_to(image_dir)
    json_file_name = rel_path.stem + ".json"
    return out_json_dir / json_file_name

def get_existing_json(out_dir: Path):
    existing = set()
    for json_file in out_dir.glob("*.json"):
        existing.add(json_file.stem)
    return existing



def get_rel_path(path: Path):
    try: 
        return path.relative_to(PROJECT_ROOT)
    except ValueError:
        return path.resolve().as_posix()


def iter_images(root: Path, exts=IMG_EXTS, out_dir: Path=None):
    exts = set(e.lower() for e in exts)
    stack = [str(root)]
    existing = get_existing_json(out_dir)
    while stack:
        d = stack.pop()
        with os.scandir(d) as it:
            for entry in it:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(entry.path)
                elif entry.is_file(follow_symlinks=False):
                    suf = Path(entry.name).suffix.lower()
                    if suf in exts:
                        if image_path_to_json_path(Path(entry.path), out_dir, root).stem in existing:
                            print("Skipping existing:", entry.path)
                            continue
                        yield Path(entry.path)

def batched(iterable, n):
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch

def write_json(out_file: Path, obj: dict):
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def run_hrnet_on_folder(image_dir: Path, out_json_dir: Path, device="cuda", det_mode="top-down", batch_size=256, io_workers=4):
    out_json_dir.mkdir(parents=True, exist_ok=True)

    if det_mode == "whole_image":
        inferencer = MMPoseInferencer(pose2d=HRNET_CFG, det_model="whole_image", device=device)
    else:
        inferencer = MMPoseInferencer(pose2d=HRNET_CFG, det_model=DET_CFG, device=device,det_cat_ids=0 )

    print("detector is None?", inferencer.inferencer.detector is None)

    paths = iter_images(image_dir, exts=IMG_EXTS, out_dir=out_json_dir)

    with ThreadPoolExecutor(max_workers=io_workers) as pool:
        print("Starting inference...")
        futures = []
        for batch_paths in batched(paths, batch_size):
            batch_paths = [p.resolve() for p in batch_paths]
            batch_rel = [str(get_rel_path(p)) for p in batch_paths]
            batch_abs = [str(p) for p in batch_paths]
            bad_log = out_json_dir / "bad_images.txt"
            result = inferencer(batch_abs, return_vis=False)

            for rel_path , res in zip(batch_rel, result):

                preds = res["predictions"][0]
                
                output = {
                    "image_path": rel_path,
                    "predictions": []
                }

                for pid, p in enumerate(preds):
                    output["predictions"].append({
                        "person_id": pid,
                        "bbox": p["bbox"],
                        "bbox_score": float(p["bbox_score"]),
                        "keypoints": p["keypoints"],
                        "keypoint_scores": p["keypoint_scores"]
                    })

                out_file = out_json_dir / (Path(rel_path).stem + ".json")
                futures.append(pool.submit(write_json, out_file, output))
            print(f"Processed batch of {len(batch_paths)} images.")

        for fu in futures:
            fu.result()

        


def run_only_one(device="cuda", image_path:str=None):
    print(inspect.signature(MMPoseInferencer))
    inferencer = MMPoseInferencer(pose2d=HRNET_CFG, det_model=DET_CFG, device=device,det_cat_ids=0 )

    for res in inferencer(image_path, return_vis=False): 
        preds = res["predictions"][0] if "predictions" in res else res
        print("num persons:", len(preds))
        for i,p in enumerate(preds[:5]):
            print(i, "bbox=", p.get("bbox"), "bbox_score=", p.get("bbox_score"))

def test_bbox(image_path:str=None):
    img = image_path
    det = DetInferencer(model="rtmdet_m_8xb32-300e_coco", device="cuda")

    out = det(img, return_vis=False)
    pred = out["predictions"][0]

    print("num det boxes:", len(pred["bboxes"]))
    for b, s, l in zip(pred["bboxes"][:10], pred["scores"][:10], pred["labels"][:10]):
        print("label", int(l), "score", float(s), "bbox", b)




def order_and_normalize_to_coco18(coco17dir: Path, out_coco18dir: Path, conf_thr=0.1):
    out_coco18dir.mkdir(parents=True, exist_ok=True)

    for json_file in coco17dir.glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)

        # print(data)
        for person in data["predictions"]:

            coco18 = convert_2d_coco17_to_coco18(person, conf_thr)
            coco18["image_path"] = data["image_path"]
            coco18["person_id"] = person["person_id"]
            
            file_name = json_file.stem + f"_person{person['person_id']}.json" 
            out_file = out_coco18dir / file_name
            with open(out_file, "w") as f:
                json.dump(coco18, f, indent=2)


if __name__ == "__main__":
    POSE_2D_JSON_PATH = os.getenv("POSE_2D_JSON_PATH")
    POSE_2D_COCO17_JSON_PATH = os.getenv("POSE_2D_COCO17_JSON_PATH")
    IMAGE_PATH = os.getenv("IMAGE_PATH")
    
    run_hrnet_on_folder(
        image_dir=Path(IMAGE_PATH),
        out_json_dir=Path(POSE_2D_COCO17_JSON_PATH),
        device="cuda",
        det_mode="top-down",
    )

    order_and_normalize_to_coco18(
        coco17dir=Path(POSE_2D_COCO17_JSON_PATH),
        out_coco18dir=Path(POSE_2D_JSON_PATH),
        conf_thr=0.3,)

    # run_only_one(device="cuda")
    # test_bbox()