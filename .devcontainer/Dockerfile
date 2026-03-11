FROM mcr.microsoft.com/devcontainers/python:1-3.11-bookworm

RUN rm -f /etc/apt/sources.list.d/yarn.list /etc/apt/sources.list.d/yarn.sources \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git \
        build-essential \
        cmake \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libxrandr2 \
        libxinerama1 \
        libxcursor1 \
        libxi6 \
        swig \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/eilab-gt/NovGrid.git
RUN cd NovGrid && pip install -e .

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
