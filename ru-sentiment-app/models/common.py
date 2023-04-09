"""Common models."""
from __future__ import annotations

import os
import pickle
from pathlib import Path

from loguru import logger
from transformers import (
    VisionEncoderDecoderModel,
    ViTFeatureExtractor,
    AutoTokenizer,
    BertTokenizer,
    BertForSequenceClassification,
    ViTForImageClassification,
)

serialized_model_path = Path(__file__).parent.resolve() / "serialized"

GPT_MODEL = "tuman/vit-rugpt2-image-captioning"
BERT_MODEL = "DeepPavlov/rubert-base-cased"
VIT_MODEL = "google/vit-base-patch16-224"

TEXT_BERT_OUTPUT = 32
VISUAL_VIT_OUTPUT = 64
VISUAL_BERT_OUTPUT = 64


if bool(os.environ.get("DESERIALIZE_MODEL", False)):
    logger.info("Deserialize huggingface models.")
    logger.info(f"Model path: {serialized_model_path}")
    with open(serialized_model_path / "image_caption_generate_model.pkl", "rb") as model_file:
        logger.info("Deserialize image_caption_generate_model.")
        image_caption_model: VisionEncoderDecoderModel = pickle.load(model_file)

    with open(serialized_model_path / "image_caption_feature_extractor.pkl", "rb") as feature_extractor_file:
        logger.info("Deserialize image_caption_feature_extractor.")
        image_caption_feature_extractor: ViTFeatureExtractor = pickle.load(feature_extractor_file)

    with open(serialized_model_path / "image_caption_tokenizer.pkl", "rb") as tokenizer_file:
        logger.info("Deserialize image_caption_tokenizer.")
        image_caption_tokenizer: AutoTokenizer = pickle.load(tokenizer_file)

    with open(serialized_model_path / "bert_tokenizer.pkl", "rb") as bert_tokenizer_file:
        logger.info("Deserialize bert_tokenizer.")
        bert_tokenizer: BertTokenizer = pickle.load(bert_tokenizer_file)

    with open(serialized_model_path / "text_bert_model.pkl", "rb") as text_bert_model_file:
        logger.info("Deserialize text_bert_model.")
        bert_model: BertForSequenceClassification = pickle.load(text_bert_model_file)

    with open(serialized_model_path / "vit_feature_extractor.pkl", "rb") as vit_feature_extractor_file:
        logger.info("Deserialize vit_feature_extractor.")
        vit_feature_extractor: ViTFeatureExtractor = pickle.load(vit_feature_extractor_file)

    with open(serialized_model_path / "bert_tokenizer.pkl", "rb") as bert_feature_extractor_file:
        logger.info("Deserialize bert_feature_extractor.")
        bert_feature_extractor: BertTokenizer = pickle.load(bert_feature_extractor_file)

    with open(serialized_model_path / "visual_model.pkl", "rb") as visual_model_file:
        logger.info("Deserialize visual_model.")
        visual_model: ViTForImageClassification = pickle.load(visual_model_file)

    with open(serialized_model_path / "language_model.pkl", "rb") as language_model_file:
        logger.info("Deserialize language_model.")
        language_model: BertForSequenceClassification = pickle.load(language_model_file)

else:
    logger.info("Download huggingface models.")
    logger.info("Download image_caption_model.")
    image_caption_model = VisionEncoderDecoderModel.from_pretrained(GPT_MODEL)

    logger.info("Download image_caption_feature_extractor.")
    image_caption_feature_extractor = ViTFeatureExtractor.from_pretrained(GPT_MODEL)

    logger.info("Download image_caption_tokenizer.")
    image_caption_tokenizer = AutoTokenizer.from_pretrained(GPT_MODEL)

    logger.info("Download bert_tokenizer.")
    bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

    logger.info("Download bert_model.")
    bert_model = BertForSequenceClassification.from_pretrained(
        BERT_MODEL,
        num_labels=TEXT_BERT_OUTPUT,
        output_attentions=False,
        output_hidden_states=False,
    )

    logger.info("Download vit_feature_extractor.")
    vit_feature_extractor = ViTFeatureExtractor.from_pretrained(VIT_MODEL)

    logger.info("Download bert_feature_extractor.")
    bert_feature_extractor = BertTokenizer.from_pretrained(BERT_MODEL)

    logger.info("Download visual_model.")
    visual_model = ViTForImageClassification.from_pretrained(
        VIT_MODEL,
        num_labels=VISUAL_VIT_OUTPUT,
        ignore_mismatched_sizes=True,
    )

    logger.info("Download language_model.")
    language_model = BertForSequenceClassification.from_pretrained(
        BERT_MODEL,
        num_labels=VISUAL_BERT_OUTPUT,
        ignore_mismatched_sizes=True,
    )
