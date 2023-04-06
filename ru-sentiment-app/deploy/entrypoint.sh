FROM python:3.9-slim

RUN apt-get update \
    && pip install --upgrade pip \
    && pip install pipenv \
    && pipenv install --system --deploy


RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117