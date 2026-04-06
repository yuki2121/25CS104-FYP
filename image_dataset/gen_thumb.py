from pathlib import Path
from PIL import Image
from database.db import get_all_image_bbox
import os
from dotenv import load_dotenv

load_dotenv()

SRC_ROOT = Path(os.getenv("IMAGE_PATH"))       
OUT_DIR  = Path(os.getenv("IMAGE_THUMBS_PATH"))   
MAX_SIDE = 320
QUALITY  = 85

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def crop_and_thumb(img_path: Path, out_path: Path, x1, y1, x2, y2, padding=0.3):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(img_path) as im:
        im = im.convert("RGB")
        w, h = im.size

        x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))

        left   = min(x1, x2)
        right  = max(x1, x2)
        top    = min(y1, y2)
        bottom = max(y1, y2)
        
        bw = right - left
        bh = bottom - top
        

        left   = left - (bw * padding)
        right  = right + (bw * padding)
        top    = top - (bh * padding)
        bottom = bottom + (bh * padding)


        left   = clamp(left,   0, w)
        right  = clamp(right,  0, w)
        top    = clamp(top,    0, h)
        bottom = clamp(bottom, 0, h)

        if right - left < 2 or bottom - top < 2:
            crop = im
        else:
            crop = im.crop((int(left), int(top), int(right), int(bottom)))

        crop.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
        crop.save(out_path, "JPEG", quality=QUALITY, optimize=True)


queryRes = get_all_image_bbox()

for pose_id, url, x1, y1, x2, y2 in queryRes:
    src = SRC_ROOT / url                    
    dst = OUT_DIR / f"{pose_id}.jpg"        
    crop_and_thumb(src, dst, x1, y1, x2, y2)

print("done")
