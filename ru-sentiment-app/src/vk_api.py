"""Module for VK API."""
from __future__ import annotations

import io
import os
import tempfile
import uuid
from typing import TYPE_CHECKING

import requests
from pathlib import Path
from PIL import Image
from fastapi.exceptions import HTTPException

from src import configurations, schemas
from src.minio_client import Minio

if TYPE_CHECKING:
    from typing import Any, Dict, List, Tuple


class VKAPI:
    """VK API class."""

    def __init__(self, api_token: str):
        self.token = api_token
        self.vk_api_url = "https://api.vk.com/method/{method}"
        self.vk_api_version = "5.131"

    @classmethod
    def __choose_photo(cls, photo_sizes: List[Dict[str, Any]]) -> Tuple[str, str]:
        """Choose photo."""
        chosen_photo_url = None
        for photo_size in photo_sizes:
            if photo_size.get("height") == 340 and photo_size.get("width") == 510:
                chosen_photo_url = photo_size.get("url")
                break

        if chosen_photo_url is None:
            chosen_photo_url = photo_sizes[-1].get("url")

        filename = chosen_photo_url.split("?")[0].split("/")[-1]

        return chosen_photo_url, filename

    @classmethod
    def __store_image_to_minio(
        cls,
        bucket: str,
        key: str,
        photo_url: str,
        filename: str,
        minio: Minio,
    ) -> Image:
        """Store image to MinIO."""
        image_response = requests.get(photo_url, stream=True)
        image = Image.open(io.BytesIO(image_response.content))
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / filename
            image.save(file_path)

            with open(file_path, "rb") as image_file:
                file_size = os.fstat(image_file.fileno()).st_size
                minio.put_object(
                    bucket_name=bucket,
                    object_name=key,
                    data=image_file,
                    length=file_size,
                )
        return image

    def download_images(self, attachments: List[Dict[str, Any]], minio: Minio) -> List[schemas.DownloadedFromVKImage]:
        """Download images from VK post."""
        downloaded_images = []
        for attachment in attachments:
            if attachment.get("type") == "photo":
                photo = attachment.get("photo")
                photo_sizes = photo.get("sizes")
                photo_text = photo.get("text")
                photo_url, filename = self.__choose_photo(photo_sizes)

                bucket = configurations.MINIO_BUCKET
                key = str(uuid.uuid4())

                image = self.__store_image_to_minio(bucket, key, photo_url, filename, minio)

                downloaded_images.append(
                    schemas.DownloadedFromVKImage(
                        bucket=bucket,
                        key=key,
                        image_url=photo_url,
                        caption=photo_text,
                        filename=filename,
                        image=image,
                    )
                )

        return downloaded_images

    def download_post(self, post: Dict[str, Any], minio: Minio) -> schemas.DownloadedPostFromVK:
        """Download post from VK."""
        images = self.download_images(post.get("attachments"), minio)
        text = post.get("text")
        post_id = f"{post.get('owner_id')}_{post.get('id')}"
        return schemas.DownloadedPostFromVK(
            post_id=post_id,
            saved_images=images,
            text=text,
        )

    def get_post_by_id(self, post_id: str, minio: Minio) -> schemas.DownloadedPostFromVK:
        """Get post by id."""
        method = "wall.getById"
        url = self.vk_api_url.format(method=method)

        params = {"v": self.vk_api_version, "access_token": self.token, "posts": post_id}

        post_response = requests.post(url, params=params)

        if post_response.status_code != 200:
            raise HTTPException(status_code=post_response.status_code, detail=post_response.text)

        post = post_response.json().get("response")[0]

        return self.download_post(post, minio)

    def get_posts_by_wall(self, owner: str, minio: Minio, count: int = 10) -> List[schemas.DownloadedPostFromVK]:
        """Get post by owner."""
        method = "wall.get"
        url = self.vk_api_url.format(method=method)

        params = {
            "v": self.vk_api_version,
            "access_token": self.token,
            "domain": owner,
            "count": count,
        }

        wall_response = requests.post(url, params=params)

        if wall_response.status_code != 200:
            raise HTTPException(status_code=wall_response.status_code, detail=wall_response.text)

        posts = wall_response.json().get("response").get("items")

        saved_posts = []
        for post in posts:
            saved_posts.append(self.download_post(post, minio))

        return saved_posts
