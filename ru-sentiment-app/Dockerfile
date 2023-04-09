FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --upgrade pip

RUN pip install poetry

COPY . /app

RUN poetry config virtualenvs.create false
RUN poetry install --only main --no-interaction --no-root

RUN pip3 install torch torchvision torchaudio

ENTRYPOINT ["./entrypoint.sh"]