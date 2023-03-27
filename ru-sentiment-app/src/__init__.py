"""RuSentimentFactors service"""
from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import torch
from fastapi import FastAPI
from transformers import (
    AutoTokenizer,
    ViTFeatureExtractor,
    VisionEncoderDecoderModel,
)

from config import Config

if TYPE_CHECKING:
    from sklearn.preprocessing import LabelEncoder


configurations = Config()
app = FastAPI(
    root_path=configurations.ROOT_PATH,
    debug=configurations.DEBUG,
    title="RuSentimentFactors service",
    version="0.1.0",
    description="Service for sentiment analysis and its factors extractions.",
    docs_url="/docs",
)

from .views import *
