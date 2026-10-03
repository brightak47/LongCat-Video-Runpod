# LongCat-Video as a RunPod serverless worker.
#
# The weights are NOT in this image. The base checkpoint is 83 GB, and an image that size takes most of an hour
# to pull the first time a worker starts on a new machine -- we measured ~50 minutes on a 177 GB image. They live
# on a network volume instead, mounted at /runpod-volume, which also means the model can be updated without
# rebuilding anything. See download_weights.py for populating it.
#
# Licence: LongCat-Video is MIT.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      git wget curl ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN git clone --depth 1 https://github.com/meituan-longcat/LongCat-Video.git /app/LongCat-Video
WORKDIR /app/LongCat-Video

# flash-attn from the project's own prebuilt wheel rather than from source. Building it here takes 30-90 minutes
# of compile time against a Hub build that is not allowed to run that long, and the wheel is the same artefact.
# The fallback exists because the wheel's name encodes the exact torch/python/ABI triple: if the base image ever
# moves, the build falls back to compiling instead of failing silently on a wheel that does not apply.
ARG FLASH_WHEEL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
RUN pip install --no-cache-dir "$FLASH_WHEEL" \
 || (echo "prebuilt flash-attn wheel did not apply; compiling" \
     && MAX_JOBS=4 pip install --no-cache-dir --no-build-isolation flash-attn==2.7.4.post1)

# The project's pins, minus torch (the base image has it) and streamlit (its demo UI is not wanted here).
RUN grep -vE '^(torch==|streamlit==)' requirements.txt > /tmp/req.txt \
 && pip install --no-cache-dir -r /tmp/req.txt \
 && pip install --no-cache-dir runpod requests boto3 "huggingface_hub>=0.23,<1.0" \
 && python -c "import torch, flash_attn, diffusers, transformers; print('torch', torch.__version__, '| flash-attn', flash_attn.__version__, '| diffusers', diffusers.__version__)"

# Proven at build time rather than on the first paid request: a missing module in this import list is the whole
# difference between a worker that starts and one that fails every job with a traceback nobody sees.
RUN python -c "\
from longcat_video.pipeline_longcat_video import LongCatVideoPipeline; \
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan; \
from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel; \
from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler; \
from longcat_video.context_parallel.context_parallel_util import init_context_parallel; \
print('longcat imports ok')"

ENV CHECKPOINT_DIR=/runpod-volume/weights/LongCat-Video
COPY download_weights.py /app/LongCat-Video/download_weights.py
COPY handler.py /app/LongCat-Video/handler.py
CMD ["python", "-u", "handler.py"]
