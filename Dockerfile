# Dockerfile

FROM python:3.10-slim

RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip \
    && pip install \
        torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    && pip install \
        fastapi \
        uvicorn[standard] \
        pydantic-settings \
        transformers \
        accelerate \
        sentencepiece \
        protobuf \
        nvitop

WORKDIR /app
CMD ["python3"]
