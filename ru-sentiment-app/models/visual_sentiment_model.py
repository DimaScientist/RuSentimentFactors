"""Visual sentiment model."""
from __future__ import annotations

import os
import pickle
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from models.common import (
    language_model,
    visual_model,
    vit_feature_extractor,
    bert_feature_extractor,
    VISUAL_VIT_OUTPUT,
    VISUAL_BERT_OUTPUT,
)

if TYPE_CHECKING:
    from typing import List, Tuple

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

BATCH_SIZE = 16

BERT_MODEL = "DeepPavlov/rubert-base-cased"
VIT_MODEL = "google/vit-base-patch16-224"

FUSION_DIM = 64

DROPOUT = 0.2

MAX_SEQ_LENGTH = 128

AVAILABLE_GPU = True

NUM_CLASSES = 3

NUM_TRAIN_EPOCHS = 5

LEARNING_RATE = 2e-5
NUM_EPOCHS = 10
EPS = 1e-7

LABEL_ENCODER_PATH = "./serialized/label_encoder.pkl"

SERIALIZED_MODELS_DIR = os.path.join(os.getcwd(), "serialized")
VISUAL_SENTIMENT_MODEL_CONFIG_PATH = os.path.join(SERIALIZED_MODELS_DIR, "sentiment_visual_model.cfg")

vit_feature_extractor = vit_feature_extractor
bert_feature_extractor = bert_feature_extractor


def transform_sentences(sentences: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Transform sentences to vectors."""
    input_ids = []
    attention_masks = []
    for sentence in tqdm(sentences):
        encoded_dict = bert_feature_extractor.encode_plus(
            sentence,
            add_special_tokens=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids.append(encoded_dict.get("input_ids"))
        attention_masks.append(encoded_dict.get("attention_mask"))

    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)


def transform_images(image_paths: List[str], image_feature_extractor) -> torch.Tensor:
    """Extract features from image."""
    pixel_values = []
    for image_path in tqdm(image_paths):
        pil_image = Image.open(image_path)
        pixel_value = image_feature_extractor(pil_image, return_tensors="pt").get("pixel_values")
        pixel_values.append(pixel_value)
    return torch.cat(pixel_values, dim=0)


def f1_weighted_score(preds: torch.Tensor, labels: torch.Tensor):
    pred_numpy = preds.cpu().detach().numpy()
    labels_numpy = labels.cpu().detach().numpy()
    pred_flatten = np.argmax(pred_numpy, axis=1).flatten()
    labels_flatten = np.argmax(labels_numpy, axis=1).flatten()
    return f1_score(
        labels_flatten.astype(np.int32),
        pred_flatten.astype(np.int32),
        average="weighted",
    )


def train(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optim: optim.Optimizer,
    loss_func: torch.nn.Module,
    warmup_scheduler,
) -> Tuple[float, float]:
    loss = 0
    accuracy = 0
    model.train()
    for input_ids, attention_mask, image_features, sentiment in dataloader:
        optim.zero_grad()

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        image_features = image_features.to(device)

        sentiment = sentiment.to(device)

        preds = model(input_ids, attention_mask, image_features)

        loss_current = loss_func(preds.to(device), sentiment)

        accuracy_curr = f1_weighted_score(preds, sentiment)

        loss_current.backward()
        optim.step()
        warmup_scheduler.step()

        loss += loss_current.item()
        accuracy += accuracy_curr.item()
    return loss / len(dataloader), accuracy / len(dataloader)


class MultimodalVisualSentimentModel(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        language_module,
        visual_module,
        language_model_dim: int,
        visual_model_dim: int,
        fusion_dim: int,
        dropout_prob: float,
    ):
        super().__init__()
        self.language_module = language_module
        self.visual_module = visual_module
        self.fusion = torch.nn.Linear(
            in_features=(language_model_dim + visual_model_dim),
            out_features=fusion_dim,
        )
        self.dropout = torch.nn.Dropout(p=dropout_prob)
        self.classifier = torch.nn.Linear(in_features=fusion_dim, out_features=num_classes)
        self.softmax = torch.nn.Softmax()
        self.relu = torch.nn.ReLU()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_features: torch.Tensor,
    ):
        language_output = self.language_module(input_ids, attention_mask=attention_mask).logits
        visual_output = self.visual_module(image_features).logits

        combined_output = torch.cat([language_output, visual_output], dim=1)

        fused_output = self.dropout(self.fusion(combined_output))

        model_logits = self.softmax(self.classifier(fused_output))

        return model_logits


visual_model = visual_model
language_model = language_model

visual_sentiment_model = MultimodalVisualSentimentModel(
    num_classes=NUM_CLASSES,
    language_module=language_model,
    visual_module=visual_model,
    language_model_dim=VISUAL_BERT_OUTPUT,
    visual_model_dim=VISUAL_VIT_OUTPUT,
    fusion_dim=FUSION_DIM,
    dropout_prob=DROPOUT,
)

if __name__ == "__main__":
    torch.cuda.empty_cache()

    if torch.cuda.is_available() and AVAILABLE_GPU:
        device = torch.device("cuda")
        print(f"Available GPU devices: {torch.cuda.device_count()}")
        print(f"Used GPU device: {torch.cuda.get_device_name()}")
        print(torch.cuda.memory_summary())
    else:
        print("There are no GPU available")
        device = torch.device("cpu")

    image_data_info_path = input("Image data info path: ")
    image_parent_path = input("Image data directory: ")

    image_info_df = pd.read_csv(image_data_info_path)
    image_info_df["image_path"] = image_info_df["image_path"].apply(lambda x: f"{image_parent_path}/" + str(x))

    with open(LABEL_ENCODER_PATH, "rb") as label_encoder_file:
        label_encoder: LabelEncoder = pickle.load(label_encoder_file)

    image_info_df["label"] = label_encoder.transform(image_info_df["sentiment"])

    input_ids, attention_masks = transform_sentences(image_info_df["clean_caption"].to_list())
    image_features = transform_images(image_info_df["image_path"].to_list(), vit_feature_extractor)
    labels = F.one_hot(torch.tensor(image_info_df["label"].to_list()).to(torch.int64)).to(torch.float64)

    tensor_dataset = TensorDataset(input_ids, attention_masks, image_features, labels)

    dataloader = DataLoader(
        tensor_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    NUM_WARMUP_STEPS = 0
    NUM_TRAINING_STEPS = len(dataloader) * NUM_TRAIN_EPOCHS

    result_multimodal_vision_model = MultimodalVisualSentimentModel(
        num_classes=NUM_CLASSES,
        language_module=language_model,
        visual_module=visual_model,
        language_model_dim=VISUAL_BERT_OUTPUT,
        visual_model_dim=VISUAL_VIT_OUTPUT,
        fusion_dim=FUSION_DIM,
        dropout_prob=DROPOUT,
    )
    result_multimodal_vision_model.to(device)

    optimizer = optim.RMSprop(
        result_multimodal_vision_model.parameters(recurse=True),
        lr=LEARNING_RATE,
        eps=EPS,
    )
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=NUM_WARMUP_STEPS,
        num_training_steps=NUM_TRAINING_STEPS,
    )

    for _ in tqdm(range(NUM_TRAIN_EPOCHS)):
        loss_value, accuracy_value = train(
            result_multimodal_vision_model,
            dataloader,
            optimizer,
            loss_function,
            scheduler,
        )
        print(f"\nLoss: {loss_value}. Accuracy: {accuracy_value}\n")

    result_multimodal_vision_model.to("cpu")
    torch.save(
        result_multimodal_vision_model.state_dict(),
        "./serialized/sentiment_visual_model.cfg",
    )
