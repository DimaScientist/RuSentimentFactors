# RuSentimentApp

Applications is for sentiment analysis and feature extractions.

## Technologies:

* __Python 3.9__
* __FastAPI__
* __Docker__
* __ClickHouse__
* __MinIO__
* __PyTorch__

## Local running

1. Install *Python 3.9*.
2. Install *poetry* follow command:

```bash
pip install poetry
```

3. Install dependencies:

```bash
poetry install 
```

4. Install *PyTorch*:

* Linux

```bash
pip3 install torch torchvision torchaudio 
```

* Windows (Cuda 11.7)

```shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

6. Set environment variables in `.env` and set variable:

* Linux:

```bash
export LOAD_ENV=true
export DESERIALIZE_MODEL=false
```

* Windows:

```shell
$env:LOAD_ENV=true
$env:DESERIALIZE_MODEL=false
```

7. Run services `ClickHouse` and `MinIO`:

```bash
docker compose -f docker-compose.dev.yml up
```

8. Run `run.py`:

```bash
python run.py
```

## Deploy

1. Create `VK_TOKEN` from VK API.
2. Set environment variables in `.env`.
3. Up docker service:

```bash
docker compose -f docker-compose.yml up --build
```