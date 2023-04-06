"""RuSentimentFactors service"""
from __future__ import annotations

from fastapi import FastAPI

from config import configurations


app = FastAPI(
    root_path=configurations.ROOT_PATH,
    debug=configurations.DEBUG,
    title="RuSentimentFactors service",
    version="0.1.0",
    description="Service for sentiment analysis and its factors extractions.",
    docs_url="/docs",
)

from .views import *
