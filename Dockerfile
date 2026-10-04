# LongCat-Video as a RunPod serverless worker, fp8 on a 24 GB card.
#
# The point of this build is the card it runs on. The official pipeline needs an 80 GB GPU, and on RunPod every
# 80 GB card is Low stock in every datacentre that also supports network volumes -- so jobs queue for hardware
# rather than running. Kijai's fp8 conversion of the same checkpoint is 15.5 GB against 27.2 GB at bf16, which
# fits a 4090 alongside its activations, and the 4090 pool has real capacity.
#
# fp8_e4m3fn is a native tensor-core format on Ada (compute 8.9), so this is not only smaller: WanVideoWrapper's
# `_fast` quantization modes do the matmul in fp8. That is the difference between this and the INT8 path in the
# upstream repo, which dequantizes to bf16 on every forward and therefore saves memory while costing speed.
#
# The weights are NOT in this image, which is a correction rather than the original plan. Baking them in removed
# the need for a network volume and looked like the whole operational win: a worker could start wherever a 4090
# was free. In practice the build never produced an image at all -- a build step downloading 24 GB from
# HuggingFace does not finish inside the Hub's limits, and the endpoint came up with `imageName: (none)` and
# workers stuck in `initializing` forever. So the weights moved to a volume, populated once with
# `mode: "setup"`, and this image went back to being small.
#
# The volume pins the endpoint to one datacentre. Make it EU-RO-1, which carries the only 4090s above `Low`
# stock found anywhere, so most of the availability this was reaching for survives the pinning.
#
# Licences: LongCat-Video is MIT. ComfyUI is GPL-3.0 and is used here as an unmodified upstream program.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

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
# turns a rename into a clear error on boot instead of a confusing failure mid-job.
#
# huggingface_hub is deliberately NOT capped below 1.0. That cap is right in the 80 GB build, whose MuseTalk-era
# transformers requires it, and wrong here: this stack's diffusers 0.40 and transformers 5.18 need hub >= 1.23,
# and pinning under it downgrades the hub and breaks `import transformers` with "cannot import name
# 'ResolvedRevision'". Only hf_hub_download is used in this worker, which every version provides.
RUN git clone --depth 1 https://github.com/kijai/ComfyUI-WanVideoWrapper.git custom_nodes/ComfyUI-WanVideoWrapper \
 && git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes.git custom_nodes/ComfyUI-KJNodes \
 && git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git custom_nodes/ComfyUI-VideoHelperSuite \
 && pip install --no-cache-dir \
      -r custom_nodes/ComfyUI-WanVideoWrapper/requirements.txt \
      -r custom_nodes/ComfyUI-KJNodes/requirements.txt \
      -r custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt \
 && pip install --no-cache-dir runpod requests boto3 huggingface_hub

# Where ComfyUI looks for models. extra_model_paths.yaml is the supported mechanism and survives an upgrade,
# which moving or symlinking the models directory would not.
COPY extra_model_paths.yaml /ComfyUI/extra_model_paths.yaml

# Proven at build time rather than on the first paid request. The import list is the important part: a
# dependency conflict is something pip reports as a warning and then exits 0 on, so a downgraded
# huggingface_hub silently broke transformers and peft until this line started catching it.
RUN ffmpeg -hide_banner -h encoder=libx264 > /dev/null \
 && python -c "import torch; print('torch', torch.__version__, '| fp8 dtype', torch.float8_e4m3fn)" \
 && python -c "import yaml, pathlib; print('model paths:', yaml.safe_load(pathlib.Path('/ComfyUI/extra_model_paths.yaml').read_text()))" \
 && python -c "import transformers, peft, diffusers, accelerate, gguf, cv2, sentencepiece, huggingface_hub; print('transformers', transformers.__version__, '| diffusers', diffusers.__version__, '| hub', huggingface_hub.__version__)"

ENV COMFY_DIR=/ComfyUI
COPY handler.py /handler.py
CMD ["python", "-u", "/handler.py"]
