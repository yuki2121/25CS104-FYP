from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .schemas import PoseSearchRequest, PoseSearchResponse
from shared.keypoints_manipulation import normalize_coco18_3d, pose_3d_to_vector
from database.db import  get_result
import hashlib, numpy as np

from fastapi.staticfiles import StaticFiles
from pathlib import Path

import os
from dotenv import load_dotenv

load_dotenv()

DATASET_DIR = os.getenv("IMAGE_DATASET_PATH") 
DATABASE_URL = os.getenv("DATABASE_URL")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[DATABASE_URL], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/dataset", StaticFiles(directory=str(DATASET_DIR)), name="dataset") 
print(f"Mounted /dataset to {DATASET_DIR}")

@app.get("/api/health")
def health():
    return {"ok": True}

@app.post("/api/search", response_model=PoseSearchResponse)
def search_pose(req: PoseSearchRequest):
    norm_xy, _ = normalize_coco18_3d(req.keypoints2d, req.score)
    vec = pose_3d_to_vector(norm_xy, req.score)

    topk = get_result(vec, req.limit, req.offset)

    return {"topK": topk}

