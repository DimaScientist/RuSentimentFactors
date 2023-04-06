"""Minio related stuff."""
from __future__ import annotations

import io
import uuid
from typing import TYPE_CHECKING

import minio

from config import configurations
from src.schemas import SavedMinioImage

if TYPE_CHECKING:
    from fastapi import UploadFile
    from typing import List, Optional, Iterator


class Minio(minio.Minio):
    """Minio client."""

    def save_images(
        self,
        images: List[UploadFile],
        captions: List[str],
        image_urls: Optional[List[str]] = None,
    ) -> List[SavedMinioImage]:
        """Save images in MinIO."""
        bucket = configurations.MINIO_BUCKET
        saved_images = []

        for i in range(len(images)):
            key = str(uuid.uuid4())

            image: UploadFile = images[i]
            data = image.file.read()
            filename = image.filename

            self.put_object(
                bucket_name=bucket,
                object_name=key,
                length=len(data),
                data=io.BytesIO(data),
                content_type=image.content_type,
            )

            image_data = {
                "bucket": bucket,
                "key": key,
                "caption": captions[i],
                "filename": filename,
            }
            if image_urls:
                image_data["image_url"] = image_urls[i]

            saved_images.append(SavedMinioImage(**image_data))

        return saved_images


def get_minio() -> Iterator[Minio]:
    """FastAPI dependency for minio connection."""
    minio_connection = Minio(
        configurations.MINIO_HOST,
        configurations.MINIO_ROOT_USER,
        configurations.MINIO_ROOT_PASSWORD,
        secure=False,
    )
    yield minio_connection
