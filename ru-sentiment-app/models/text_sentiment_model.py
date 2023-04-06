"""Text sentiment model."""
from __future__ import annotations

import os
import pickle
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from torch import optim
from torch.nn import functional as F
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)
from tqdm import tqdm
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)

if TYPE_CHECKING:
    from typing import Tuple

BERT_MODEL = "DeepPavlov/rubert-base-cased"
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32
N_SAMPLES = 100000
NUM_CLASSES = 3
BERT_OUTPUT = 32

LEARNING_RATE = 2e-6
EPS = 1e-8
NUM_EPOCHS = 3

SERIALIZED_MODELS_DIR = os.path.join(os.getcwd(), "serialized")
TEXT_SENTIMENT_MODEL_CONFIG_PATH = os.path.join(SERIALIZED_MODELS_DIR, "sentiment_text_model.cfg")
LABEL_ENCODER_PATH = os.path.join(SERIALIZED_MODELS_DIR, "label_encoder.pkl")


class TextSentimentModel(torch.nn.Module):
    def __init__(self, num_classes: int, language_model, language_model_dim: int):
        super().__init__()
        self.language_model = language_model
        self.classifier = torch.nn.Linear(
            in_features=language_model_dim,
            out_features=num_classes,
        )
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        language_output = self.language_model(
            input_ids,
            token_type_ids=None,
            attention_mask=attention_mask,
        ).logits
        logits = self.softmax(self.classifier(language_output))
        return logits


bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
bert_model = BertForSequenceClassification.from_pretrained(
    BERT_MODEL,
    num_labels=BERT_OUTPUT,
    output_attentions=False,
    output_hidden_states=False,
)
text_sentiment_model = TextSentimentModel(
    num_classes=NUM_CLASSES,
    language_model=bert_model,
    language_model_dim=BERT_OUTPUT,
)


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
    warmup_scheduler=None,
) -> Tuple[float, float]:
    loss = 0
    accuracy = 0
    model.train()
    for input_ids, attention_mask, sentiment in dataloader:
        optim.zero_grad()

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        sentiment = sentiment.to(device)

        logits = model(input_ids, attention_mask)

        loss_current = loss_func(logits.to(device), sentiment)

        accuracy_curr = f1_weighted_score(logits, sentiment)

        loss_current.backward()
        optim.step()
        if warmup_scheduler:
            warmup_scheduler.step()

        loss += loss_current.item()
        accuracy += accuracy_curr.item()
    return loss / len(dataloader), accuracy / len(dataloader)


def transform_sentences(sentences: list) -> tuple:
    input_ids = []
    attention_masks = []
    for sentence in tqdm(sentences):
        encoded_dict = bert_tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids.append(encoded_dict["input_ids"])
        attention_masks.append(encoded_dict["attention_mask"])
    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)


if __name__ == "__main__":
    if not os.path.exists(SERIALIZED_MODELS_DIR):
        raise Exception(f"{SERIALIZED_MODELS_DIR} is not exists.")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Available GPU devices: {torch.cuda.device_count()}")
        print(f"Used GPU device: {torch.cuda.get_device_name()}")
        print(torch.cuda.memory_summary())
    else:
        print("There are no GPU available")
        device = torch.device("cpu")

    torch.cuda.empty_cache()

    text_data_path = input("Text data path: ")
    text_data = pd.read_csv(text_data_path)
    text_data = text_data.drop_duplicates(subset=["text"])

    text_data = text_data.dropna(subset=["text"]).sample(frac=1).reset_index(drop=True)

    label_encoder = LabelEncoder()
    label_encoder.fit(text_data["sentiment"])

    text_data["sentiment"] = label_encoder.transform(text_data["sentiment"])

    if N_SAMPLES:
        text_data = text_data.sample(N_SAMPLES)

    input_ids, attention_masks = transform_sentences(text_data["text"].to_list())
    labels = F.one_hot(torch.tensor(text_data["sentiment"].to_list()).to(torch.int64)).to(torch.float64)

    tensor_dataset = TensorDataset(input_ids, attention_masks, labels)

    dataloader = DataLoader(
        tensor_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    result_text_sentiment_model = text_sentiment_model
    result_text_sentiment_model.to(device)

    NUM_WARMUP_STEPS = 0
    NUM_TRAINING_STEPS = len(dataloader) * NUM_EPOCHS

    optimizer = torch.optim.AdamW(result_text_sentiment_model.parameters(), lr=LEARNING_RATE, eps=EPS)
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=NUM_WARMUP_STEPS,
        num_training_steps=NUM_TRAINING_STEPS,
    )

    for _ in tqdm(range(NUM_EPOCHS)):
        loss_value, accuracy_value = train(
            result_text_sentiment_model,
            dataloader,
            optimizer,
            loss_function,
            scheduler,
        )
        print(f"\nLoss: {loss_value}. Accuracy: {accuracy_value}\n")

    result_text_sentiment_model.to("cpu")
    torch.save(
        result_text_sentiment_model.state_dict(),
        "./serialized/sentiment_text_model.cfg",
    )

    with open(LABEL_ENCODER_PATH, "wb+") as label_encoder_file:
        pickle.dump(label_encoder, label_encoder_file)
