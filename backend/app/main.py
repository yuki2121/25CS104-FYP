from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .schemas import PoseSearchRequest, PoseSearchResponse
from shared.keypoints_manipulation import normalize_coco18_3d, pose_3d_to_vector
from database.db import  get_result
import hashlib, numpy as np
from fastapi import HTTPException
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from fastapi.responses import StreamingResponse
from google.cloud import storage
import os
from dotenv import load_dotenv
from google.api_core.exceptions import NotFound
import numpy as np

import io
import mimetypes

load_dotenv()


DATABASE_URL = os.getenv("DATABASE_URL")
FRONTEND_URL = os.getenv("FRONTEND_URL")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GCS_BUCKET = os.getenv("GCS_BUCKET")
storage_client = storage.Client()

@app.get("/api/health")
def health():
    return {"ok": True}

@app.post("/api/search", response_model=PoseSearchResponse)
def search_pose(req: PoseSearchRequest):
    raw_3d = np.asarray(req.keypoints2d, dtype=np.float32)

    raw_3d = raw_3d * np.array([1, 1, -1], dtype=np.float32)

    norm_3d, _ = normalize_coco18_3d(raw_3d.tolist(), req.score)
    vec = pose_3d_to_vector(norm_3d, req.score)

    candidate_limit = max(1000, req.limit + req.offset)

    candidates = get_result(vec, candidate_limit, 0)

    reranked = rerank_candidates_holistic(
        candidates=candidates,
        query_3d=np.asarray(norm_3d, dtype=np.float32),
    )

    topk = reranked[req.offset:req.offset + req.limit]

    return {"topK": topk}

@app.get("/api/image/{object_path:path}")
def get_image(object_path: str):
    bucket = storage_client.bucket(GCS_BUCKET)
    blob = bucket.blob(object_path)

    try:
        data = blob.download_as_bytes()
    except NotFound:
        raise HTTPException(status_code=404, detail="Image not found")

    content_type = (
        blob.content_type
        or mimetypes.guess_type(object_path)[0]
        or "application/octet-stream"
    )

    return StreamingResponse(
        io.BytesIO(data),
        media_type=content_type,
        headers={"Cache-Control": "public, max-age=3600"},
    )
    
BODY_IDX = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])

PARTS = {
    "torso": [1, 2, 5, 8, 11],
    "right_arm": [2, 3, 4],
    "left_arm": [5, 6, 7],
    "right_leg": [8, 9, 10],
    "left_leg": [11, 12, 13],
}

PART_WEIGHTS = {
    "torso": 1.40,
    "right_arm": 1.10,
    "left_arm": 1.10,
    "right_leg": 1.00,
    "left_leg": 1.00,
}


def vector_to_np(v):
    if isinstance(v, np.ndarray):
        return v.astype(np.float32)

    if isinstance(v, (list, tuple)):
        return np.asarray(v, dtype=np.float32)

    if isinstance(v, str):
        s = v.strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        return np.fromstring(s, sep=",", dtype=np.float32)

    return np.asarray(v, dtype=np.float32)


def pose_vec_to_3d(v):
    arr = vector_to_np(v)

    if arr.size == 54:
        return arr.reshape(18, 3)

    if arr.size == 72:
        return arr.reshape(18, 4)[:, :3]

    return None


def normalize_pose_for_rerank(pose):
    pose = np.asarray(pose, dtype=np.float32)

    neck = pose[1]
    r_hip = pose[8]
    l_hip = pose[11]
    pelvis = (r_hip + l_hip) * 0.5

    center = pelvis
    pose = pose - center

    scale = np.linalg.norm(neck - pelvis)

    if scale < 1e-6:
        scale = np.sqrt(np.mean(np.sum(pose[BODY_IDX] ** 2, axis=1)))

    pose = pose / (scale + 1e-8)

    return pose


def part_distance(q, c, idxs):
    q_part = q[idxs]
    c_part = c[idxs]

    d = np.linalg.norm(q_part - c_part, axis=1)

    return float(np.mean(d))


def holistic_pose_distance(q, c):
    if q is None or c is None:
        return 999.0

    q = normalize_pose_for_rerank(q)
    c = normalize_pose_for_rerank(c)

    joint_d = np.linalg.norm(q[BODY_IDX] - c[BODY_IDX], axis=1)

    mean_d = float(np.mean(joint_d))
    p80_d = float(np.percentile(joint_d, 80))
    max_d = float(np.max(joint_d))

    weighted_part_sum = 0.0
    weight_sum = 0.0

    for name, idxs in PARTS.items():
        w = PART_WEIGHTS[name]
        d = part_distance(q, c, idxs)
        weighted_part_sum += w * d
        weight_sum += w

    part_d = weighted_part_sum / weight_sum

    final_d = (
        0.35 * mean_d +
        0.35 * p80_d +
        0.15 * max_d +
        0.15 * part_d
    )

    return float(final_d)


def rerank_candidates_holistic(candidates, query_3d):
    temp = []

    for item in candidates:
        cand_3d = pose_vec_to_3d(item.get("pose_3d_vec"))

        dist_holistic = holistic_pose_distance(query_3d, cand_3d)
        dist_hnsw = float(item.get("dist", 1.0))

        new_item = dict(item)
        new_item["dist_hnsw"] = dist_hnsw
        new_item["dist_holistic"] = dist_holistic

        temp.append(new_item)

    temp.sort(key=lambda x: x["dist_hnsw"])

    if not temp:
        return []

    best_hnsw = temp[0]["dist_hnsw"]

    close_pool = [x for x in temp if x["dist_hnsw"] <= best_hnsw + 0.15]
    far_pool = [x for x in temp if x["dist_hnsw"] > best_hnsw + 0.15]

    close_pool.sort(key=lambda x: x["dist_holistic"])
    far_pool.sort(key=lambda x: x["dist_hnsw"])

    reranked = close_pool + far_pool

    for item in reranked:
        item["dist"] = item["dist_holistic"]
        item.pop("pose_3d_vec", None)

    return reranked