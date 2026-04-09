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
    norm_xy, _ = normalize_coco18_3d(req.keypoints2d, req.score)
    vec = pose_3d_to_vector(norm_xy, req.score)

    topk = get_result(vec, req.limit, req.offset)

    return {"topK": topk}

@app.get("/api/image/{object_path:path}")
def get_image(object_path: str):
    bucket = storage_client.bucket(GCS_BUCKET)
    blob = bucket.blob(object_path)

    if not blob.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    data = blob.download_as_bytes()
    content_type = blob.content_type or mimetypes.guess_type(object_path)[0] or "application/octet-stream"

    return StreamingResponse(io.BytesIO(data), media_type=content_type)