FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_CACHE_DIR=/root/.cache/pip

# OpenCV runtime deps + curl for the face-detection model + ffmpeg for
# transcoding browser-incompatible uploads (e.g. FMP4/MJPEG) to H.264
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# YuNet DNN face detector (OpenCV >= 5 dropped the Haar CascadeClassifier)
RUN mkdir -p /app/models \
    && curl --fail --show-error --location --retry 3 --retry-delay 2 \
       -o /app/models/face_detection_yunet_2023mar.onnx \
       https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

# CUDA runtime for the RTX 4050. The image installs the pinned CUDA torch wheels before project dependencies.
# Own layer BEFORE any project files, with a persistent pip wheel cache mount,
# so the 780 MB CUDA wheel downloads exactly once and is reused on all
# subsequent rebuilds (even when code/requirements change).
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 torchvision==0.20.1

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Bake ImageNet weights into the image so training works offline in the container
RUN python -c "import torchvision; torchvision.models.EfficientNet_B4_Weights.IMAGENET1K_V1.get_state_dict(progress=True)"

COPY pyproject.toml ./
COPY rlroinet/ ./rlroinet/
COPY tests/ ./tests/
COPY website/ ./website/

# --no-build-isolation: reuse the setuptools already in the image so this layer
# never requires network access on rebuilds (torch/CPU wheels stay cached).
RUN pip install --no-build-isolation -e .
RUN mkdir -p /app/data /app/outputs /app/uploads

EXPOSE 8000

CMD ["python", "-m", "rlroinet.train"]
