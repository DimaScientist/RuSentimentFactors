"""Generation image captions module."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

import nltk
import torch
from PIL import Image
from nltk.corpus import stopwords
from pymystem3 import Mystem

from models.common import image_caption_tokenizer, image_caption_model, image_caption_feature_extractor

if TYPE_CHECKING:
    from typing import List

LANGUAGE = "russian"
nltk.download("stopwords")
lang_stopwords = set(stopwords.words(LANGUAGE))
mystem = Mystem()


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

image_caption_model.to(device)


def clean_text(sentence: str) -> str:
    """Clean text."""
    letters_only = re.sub("[^а-яА-ЯёЁ]", " ", sentence)
    lowercase_text = letters_only.lower()
    words = lowercase_text.split()
    lemmatize_words = [mystem.lemmatize(word)[0] for word in words if word not in lang_stopwords]
    processes_string = " ".join(lemmatize_words)
    return processes_string


def predict_captions(images: List[Image]) -> List[str]:
    images_ = []
    for image in images:
        if image.mode != "RGB":
            image = image.convert(mode="RGB")

        images_.append(image)

    pixel_values = image_caption_feature_extractor(images=images_, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = image_caption_model.generate(pixel_values)

    predicted_captions = image_caption_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    predicted_captions = [clean_text(predicted_caption.strip()) for predicted_caption in predicted_captions]
    return predicted_captions
