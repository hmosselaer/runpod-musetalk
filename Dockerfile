# MuseTalk 1.5 — RunPod serverless lip-sync worker
# Contract:  input {video_base64, audio_base64}  ->  output {video: <base64 mp4>}
#
# Built on MuseTalk's own repo + install steps so we rely on their maintained
# setup rather than reinventing it. The two fiddly steps (flagged below) are the
# OpenMMLab install and the model-weight download — if the build fails, it's
# almost certainly one of those two.
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-dev \
        git wget curl ffmpeg libgl1 libglib2.0-0 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN git clone --depth 1 https://github.com/TMElyralab/MuseTalk.git /app

# Torch (CUDA 11.8) then MuseTalk's own requirements
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 \
         --index-url https://download.pytorch.org/whl/cu118 \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "huggingface_hub[cli]" hf_transfer

# ── fiddly step 1: OpenMMLab stack (version-locked) ─────────────────────────
# chumpy (pulled in by mmpose) can't build under pip's build isolation — its
# setup.py imports numpy at build time. Install it first with --no-build-isolation
# against a present numpy<2 (mmcv 2.0.1 also needs numpy<2).
RUN pip install --no-cache-dir "numpy<2" Cython setuptools wheel \
    && pip install --no-cache-dir chumpy==0.70 --no-build-isolation
RUN pip install --no-cache-dir -U openmim \
    && mim install "mmengine" "mmcv==2.0.1" "mmdet==3.1.0" "mmpose==1.1.0"

# ── fiddly step 2: model weights (~several GB into ./models) ────────────────
# MuseTalk ships a download script; name varies by version, so try both.
RUN (sh ./download_weights.sh || bash ./download_weights.sh || python download_weights.py) \
    && ls -la models

RUN pip install --no-cache-dir runpod
COPY handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]
