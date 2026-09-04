FROM ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:py3.12-latest

LABEL org.opencontainers.image.title="VideoGenerator FastVideo"
LABEL org.opencontainers.image.description="FastVideo 0.2.1 with Wan I2V requested-canvas resolution patch"

COPY docker/fastvideo/patch_wan_i2v_resolution.py /tmp/patch_wan_i2v_resolution.py

RUN /opt/venv/bin/python /tmp/patch_wan_i2v_resolution.py \
    && rm /tmp/patch_wan_i2v_resolution.py
