# syntax=docker/dockerfile:1.6
ARG BASE_IMAGE=pytorch/pytorch:2.4.0-rocm6.0
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive     PIP_NO_CACHE_DIR=1     PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1

RUN apt-get update &&     apt-get install -y --no-install-recommends ffmpeg git &&     rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY transcribe.py /app/transcribe.py

CMD ["python", "/app/transcribe.py", "--help"]
