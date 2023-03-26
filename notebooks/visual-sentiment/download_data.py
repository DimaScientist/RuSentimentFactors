from datetime import datetime
import os
from pathlib import Path
from typing import Union

from urllib3.exceptions import MaxRetryError, ConnectionError

import math
import requests
import pickle
from tqdm import tqdm

IMAGE_DIR = "./data/source_images"
API_KEY = ""

dataset_image_path = Path(IMAGE_DIR)
chosen_image_dict = {}

with open("negative", "rb+") as negative_file:
    negative_list = pickle.load(negative_file)
    chosen_image_dict["negative"] = negative_list

with open("positive", "rb+") as positive_file:
    positive_list = pickle.load(positive_file)
    chosen_image_dict["positive"] = positive_list

with open("neutral", "rb+") as neutral_file:
    neutral_list = pickle.load(neutral_file)
    chosen_image_dict["neutral"] = neutral_list

total_images = len(chosen_image_dict["negative"]) \
               + len(chosen_image_dict["positive"]) \
               + len(chosen_image_dict["neutral"])

print(f"Total images for preprocessing: {total_images}")

error_images_count = 0

for key in chosen_image_dict.keys():
    print(f"Download images from class: {key}")
    chosen_image_list = chosen_image_dict[key]

    dir_name = dataset_image_path / key

    if os.path.exists(dir_name) is False:
        os.makedirs(dir_name)

    print(dir_name.absolute().as_posix())

    for image_id in tqdm(chosen_image_list):

        try:
            image_response = requests.get(url="https://www.flickr.com/services/rest/", params={
                "method": "flickr.photos.getSizes",
                "api_key": API_KEY,
                "photo_id": f"{image_id}",
                "format": "json",
                "nojsoncallback": 1,
            })

            image_info = image_response.json()

            if image_response.status_code == 200 and image_info["stat"] == "ok":

                image_list = image_info.get("sizes").get("size")

                image = image_list[-1]
                for image_item in image_list:
                    if image_item["label"] == "Medium":
                        image = image_item
                        break

                download_image_response = requests.get(url=f"{image.get('source')}", stream=True)
                image_format = str(image.get('source')).rsplit(".")[-1]

                with open(dir_name / f"{image_id}.{image_format}", "wb+") as file_image:
                    for chunk in download_image_response:
                        file_image.write(chunk)

        except Union[MaxRetryError, ConnectionError]:
            print(f"Connection is broken: {datetime.now()}.")
            break
        except Exception:
            error_images_count += 1

print(f"Unloaded: {error_images_count} ({math.ceil(error_images_count / total_images)})")
