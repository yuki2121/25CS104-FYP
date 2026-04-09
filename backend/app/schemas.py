from pydantic import BaseModel, Field
from typing import List, Literal, Optional

class SearchResult(BaseModel):
    pose_id: str
    url: str | None = None
    bbox_top_x: float
    bbox_top_y: float
    bbox_bottom_x: float
    bbox_bottom_y: float

class PoseSearchResponse(BaseModel):
    topK: List[SearchResult]

class PoseSearchRequest(BaseModel):
    format: Literal["openpose18"]
    keypoints2d: List[List[float]]
    score: List[float]
    limit: int = 20
    offset: int = 0
    

