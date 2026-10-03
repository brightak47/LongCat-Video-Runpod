# LongCat-Video as a RunPod serverless worker, fp8 on a 24 GB card.
#
# The point of this build is the card it runs on. The official pipeline needs an 80 GB GPU, and on RunPod every
# 80 GB card is Low stock in every datacentre that also supports network volumes -- so jobs queue for hardware
# rather than running. Kijai's fp8 conversion of the same checkpoint is 15.5 GB against 27.2 GB at bf16, which
# fits a 4090 alongside its activations, and ADA_24 is the one tier with real capacity.
#
# fp8_e4m3fn is a native tensor-core format on Ada (compute 8.9), so this is not only smaller: WanVideoWrapper's
# `_fast` quantization modes do the matmul in fp8. That is the difference between this and the INT8 path in the
# upstream repo, which dequantizes to bf16 on every forward and therefore saves memory while costing speed.
#
# The weights are baked in, deliberately. They total ~24 GB, which is a large image but means no network volume:
# no datacentre pinning, no separate setup step, and a worker can start anywhere a 4090 is free. RunPod caches
# the image per machine, so the pull is paid once per host rather than once per job.
#
# Licences: LongCat-Video is MIT. ComfyUI is GPL-3.0 and is used here as an unmodified upstream program.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      git wget curl ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /ComfyUI
WORKDIR /ComfyUI
RUN pip install --no-cache-dir -r requirements.txt

# The three node packs Kijai's own LongCat workflow is built from. Pinned to nothing on purpose: these move
# quickly and the graph this worker builds is checked against the live schema at startup (see handler.py), which
# catches a rename as a clear error on boot instead of a confusing failure mid-job.
RUN git clone --depth 1 https://github.com/kijai/ComfyUI-WanVideoWrapper.git custom_nodes/ComfyUI-WanVideoWrapper \
 && git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes.git custom_nodes/ComfyUI-KJNodes \
 && git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git custom_nodes/ComfyUI-VideoHelperSuite \
 && pip install --no-cache-dir \
      -r custom_nodes/ComfyUI-WanVideoWrapper/requirements.txt \
      -r custom_nodes/ComfyUI-KJNodes/requirements.txt \
      -r custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt \
 && pip install --no-cache-dir runpod requests boto3 "huggingface_hub>=0.23,<1.0"

# Weights. Each is fetched by exact filename rather than by snapshot, so a new file appearing upstream cannot
# silently change what this image contains.
ARG HF=https://huggingface.co
RUN python - <<'PY'
from huggingface_hub import hf_hub_download
import pathlib, shutil

WANTED = [
    # (repo, filename, destination under /ComfyUI/models)
    ("Kijai/LongCat-Video_comfy", "LongCat_TI2V_comfy_fp8_e4m3fn_scaled_KJ.safetensors", "diffusion_models/LongCat"),
    # The distilled schedule: 10 steps at cfg 1.0 instead of 50 at cfg 4.0, which is what makes this affordable.
    ("Kijai/LongCat-Video_comfy", "LongCat_distill_lora_alpha64_bf16.safetensors", "loras"),
    # fp8 text encoder rather than bf16: 6.7 GB instead of 11, and it is loaded and freed per prompt anyway.
    ("Kijai/WanVideo_comfy", "umt5-xxl-enc-fp8_e4m3fn.safetensors", "text_encoders"),
    ("Kijai/WanVideo_comfy", "Wan2_1_VAE_bf16.safetensors", "vae"),
]
for repo, name, sub in WANTED:
    target = pathlib.Path("/ComfyUI/models") / sub
    target.mkdir(parents=True, exist_ok=True)
    got = hf_hub_download(repo_id=repo, filename=name)
    shutil.copy2(got, target / pathlib.Path(name).name)
    size = (target / pathlib.Path(name).name).stat().st_size / 1e9
    print(f"{name} -> {target} ({size:.1f} GB)", flush=True)
    # The HF cache holds a second copy; this image is large enough already.
    shutil.rmtree(pathlib.Path(got).parents[2], ignore_errors=True)
PY

# Proven at build time rather than on the first paid request.
RUN ls -l /ComfyUI/models/diffusion_models/LongCat /ComfyUI/models/loras /ComfyUI/models/text_encoders /ComfyUI/models/vae \
 && ffmpeg -hide_banner -h encoder=libx264 > /dev/null \
 && python -c "import torch; print('torch', torch.__version__)"

ENV COMFY_DIR=/ComfyUI
COPY handler.py /handler.py
CMD ["python", "-u", "/handler.py"]
