import datetime
import os
from google.cloud import storage

GCS_BUCKET = os.getenv("GCS_BUCKET", "ai-pose-storage")
GCS_SIGNED_URL_MINUTES = int(os.getenv("GCS_SIGNED_URL_MINUTES", "15"))

_storage_client = storage.Client()

def sign_image_url(object_name: str) -> str:
    bucket = _storage_client.bucket(GCS_BUCKET)
    blob = bucket.blob(object_name)

    return blob.generate_signed_url(
        version="v4",
        expiration=datetime.timedelta(minutes=GCS_SIGNED_URL_MINUTES),
        method="GET",
    )