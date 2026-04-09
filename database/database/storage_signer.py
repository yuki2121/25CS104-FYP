import datetime
import os
import google.auth
from google.auth.transport.requests import Request
from google.cloud import storage

GCS_BUCKET = os.getenv("GCS_BUCKET", "ai-pose-storage")
GCS_SIGNED_URL_MINUTES = int(os.getenv("GCS_SIGNED_URL_MINUTES", "15"))

_storage_client = storage.Client()

def sign_image_url(object_name: str) -> str:
    credentials, _ = google.auth.default()
    credentials.refresh(Request())

    service_account_email = getattr(
        credentials,
        "service_account_email",
        os.getenv("SERVICE_ACCOUNT_EMAIL"),
    )
    if not service_account_email:
        raise RuntimeError("SERVICE_ACCOUNT_EMAIL is not set and could not be inferred")

    bucket = _storage_client.bucket(GCS_BUCKET)
    blob = bucket.blob(object_name)

    return blob.generate_signed_url(
        version="v4",
        expiration=datetime.timedelta(minutes=GCS_SIGNED_URL_MINUTES),
        method="GET",
        service_account_email=service_account_email,
        access_token=credentials.token,
    )