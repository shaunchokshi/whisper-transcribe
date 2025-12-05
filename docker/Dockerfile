# syntax=docker/dockerfile:1.6
ARG BASE_IMAGE=rocm/pytorch:rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.9.1
FROM ${BASE_IMAGE}

# Set noninteractive to avoid tzdata prompts in CI
ENV DEBIAN_FRONTEND=noninteractive         PIP_NO_CACHE_DIR=1         PYTHONDONTWRITEBYTECODE=1         PYTHONUNBUFFERED=1

RUN apt-get update &&         apt-get install -y --no-install-recommends ffmpeg git &&         rm -rf /var/lib/apt/lists/*

# Create workspace
WORKDIR /app

# Install Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Copy app
COPY transcribe.py /app/transcribe.py

# Default command prints usage
CMD ["python", "/app/transcribe.py", "--help"]
