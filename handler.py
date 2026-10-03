"""RunPod serverless worker for LongCat-Video in fp8, driving ComfyUI + WanVideoWrapper.

Input (job["input"]):
    mode            "t2v" (default) | "i2v" | "continue"
    prompt          required for t2v and i2v
    negative_prompt optional; a default is applied
    image_url | image_b64     the first frame, for i2v
    video_url | video_b64     the clip to carry on from, for continue
    segments        how many 93-frame passes to make, default 1. Each after the first adds ~5.3 s of video
    seconds         asked-for length; converted to segments, which is easier to think in
    width, height   default 832x480, both divisible by 16
    steps, cfg, shift, seed    overrides for the distilled schedule
    blocks_to_swap  transformer blocks held on CPU, default 0 on a 24 GB card and 20 if memory is tight
    time_budget_sec stop starting new segments after this, default 2700
    output_key      destination key in the bucket
    project         echoed back for cost attribution

Output:
    {"video_url": ...} or {"video_b64": ...}, plus {"frames", "fps", "seconds", "segments_done",
    "complete", "gpu_seconds", "project"?}

Long video:
    LongCat extends a clip 93 frames at a time, conditioned on the previous segment's last 13 frames, so a
    minute is eleven passes and ten minutes is a hundred and thirteen. More than one job's worth, and this
    worker deliberately has no network volume -- that is what lets it run on any free 4090 instead of queueing
    for an 80 GB card in one datacentre. So there is nowhere on disk to leave a resume token.

    Continuation across jobs therefore goes through the bucket: a job returns its video_url, and the caller
    passes that back as `video_url` with `mode: "continue"`. Nothing is kept on the worker between jobs, which
    is also why a failure costs one chunk rather than the whole video.

Why fp8 rather than the official pipeline:
    The upstream pipeline needs 80 GB, and on RunPod every 80 GB card is Low stock in every datacentre that
    also supports volumes. Kijai's fp8 conversion is 15.5 GB against 27.2 at bf16 and fits a 4090; fp8_e4m3fn
    is a native tensor-core format on Ada, so WanVideoWrapper's `_fast` modes do the matmul in fp8 rather than
    dequantizing first. The 80 GB build is kept under official-pipeline/ as the fallback.
"""

import base64
import glob
import json
import mimetypes
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import requests
import runpod

COMFY_DIR = Path(os.environ.get("COMFY_DIR", "/ComfyUI"))
COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
INPUT_DIR = COMFY_DIR / "input"
OUTPUT_DIR = COMFY_DIR / "output"
VOLUME_DIR = Path("/runpod-volume")
INLINE_LIMIT_MB = int(os.environ.get("INLINE_LIMIT_MB", "18"))
DOWNLOAD_TIMEOUT = int(os.environ.get("DOWNLOAD_TIMEOUT", "180"))

# Properties of the checkpoint rather than choices. num_frames must satisfy (n - 1) % 4 == 0 for the VAE's
# temporal scale factor; 93 and an overlap of 13 are what the conditioning window is built around.
FPS = 15
SEGMENT_FRAMES = 93
OVERLAP = 13
NEW_FRAMES_PER_SEGMENT = SEGMENT_FRAMES - OVERLAP

# Taken from Kijai's own LongCat_TI2V workflow rather than guessed: the distilled schedule is 10 steps at
# cfg 1.0 with shift 12, on its own scheduler. Fifty steps at cfg 4 is the undistilled path and costs five
# times as much for no benefit once the LoRA is loaded.
DIT = "LongCat/LongCat_TI2V_comfy_fp8_e4m3fn_scaled_KJ.safetensors"
DISTILL_LORA = "LongCat_distill_lora_alpha64_bf16.safetensors"
TEXT_ENCODER = "umt5-xxl-enc-fp8_e4m3fn.safetensors"
VAE = "Wan2_1_VAE_bf16.safetensors"
SCHEDULER = "longcat_distill_euler"
STEPS, CFG, SHIFT = 10, 1.0, 12.0
# `_fast` does the matmul in fp8 and needs compute capability >= 8.9, which is exactly the 4000 series this
# image targets. On an older card set LONGCAT_QUANTIZATION=disabled to let the loader autoselect.
QUANTIZATION = os.environ.get("LONGCAT_QUANTIZATION", "fp8_e4m3fn_scaled_fast")
ATTENTION = os.environ.get("LONGCAT_ATTENTION", "sdpa")

DEFAULT_NEGATIVE = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, "
    "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still "
    "picture, messy background, three legs, many people in the background, walking backwards"
)

# Every node this worker builds a graph from. Checked against the live /object_info once at boot, because these
# packs move quickly and a rename should be a clear error on startup rather than a puzzling one mid-job.
REQUIRED_NODES = [
    "WanVideoModelLoader", "WanVideoVAELoader", "WanVideoTextEncodeCached", "WanVideoEmptyEmbeds",
    "WanVideoSampler", "WanVideoDecode", "WanVideoEncode", "WanVideoBlockSwap", "WanVideoLoraSelect",
    "ImageBatchExtendWithOverlap", "ImageResizeKJv2", "LoadImage", "VHS_VideoCombine",
]


class InputError(Exception):
    """Something wrong with the request rather than the worker."""


_comfy = None
_available = set()


def start_comfy():
    """
    Bring ComfyUI up once per worker and keep it.

    It loads the fp8 DiT lazily on first use and holds it, so the first job on a worker pays the model load and
    every later one is warm. Bound to localhost: nothing but this handler should be able to reach it.
    """
    global _comfy, _available
    if _comfy is not None:
        return

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _comfy = subprocess.Popen(
        ["python", "main.py", "--listen", "127.0.0.1", "--port", "8188", "--disable-auto-launch",
         "--output-directory", str(OUTPUT_DIR), "--input-directory", str(INPUT_DIR)],
        cwd=str(COMFY_DIR), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )

    deadline = time.time() + 600
    while time.time() < deadline:
        if _comfy.poll() is not None:
            raise RuntimeError(f"ComfyUI exited while starting:\n{(_comfy.stdout.read() or '')[-3000:]}")
        try:
            requests.get(f"{COMFY_URL}/system_stats", timeout=5).raise_for_status()
            break
        except requests.RequestException:
            time.sleep(2)
    else:
        raise RuntimeError("ComfyUI did not answer within ten minutes of starting")

    info = requests.get(f"{COMFY_URL}/object_info", timeout=120).json()
    _available = set(info)
    missing = [n for n in REQUIRED_NODES if n not in _available]
    if missing:
        raise RuntimeError(
            "ComfyUI started but these nodes are missing, so the graph cannot be built: "
            + ", ".join(missing)
            + ". A node pack has probably renamed something; check the handler against the installed version."
        )
    print(f"[longcat] ComfyUI ready, {len(_available)} node types available", flush=True)


def _fetch(url: str, dest: Path) -> Path:
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
        response.raise_for_status()
        with open(dest, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                handle.write(chunk)
    if dest.stat().st_size == 0:
        raise InputError(f"downloaded file is empty: {url}")
    return dest


def _stage_input(job_input: dict, kind: str) -> str:
    """
    Put a supplied image or video where ComfyUI can load it, and return the bare filename.

    ComfyUI resolves LoadImage against its own input directory and rejects paths that try to leave it, so the
    file is copied in under a generated name rather than referenced wherever it landed.
    """
    suffix = {"image": ".png", "video": ".mp4"}[kind]
    name = f"{kind}_{uuid.uuid4().hex}{suffix}"
    dest = INPUT_DIR / name
    if job_input.get(f"{kind}_url"):
        _fetch(job_input[f"{kind}_url"], dest)
    elif job_input.get(f"{kind}_b64"):
        dest.write_bytes(base64.b64decode(job_input[f"{kind}_b64"]))
    else:
        return None
    return name


def segments_for(job_input: dict) -> int:
    """
    How many passes this request is asking for.

    `seconds` is what a caller thinks in, so it is converted here. The first pass produces a whole 93-frame
    segment; each one after it contributes only the 80 frames that are not overlap.
    """
    if job_input.get("segments") is not None:
        return max(1, int(job_input["segments"]))
    seconds = job_input.get("seconds")
    if not seconds:
        return 1
    wanted = float(seconds) * FPS
    extra = max(0, (wanted - SEGMENT_FRAMES) / NEW_FRAMES_PER_SEGMENT)
    return 1 + int(extra + 0.999)


def build_graph(job_input: dict, segments: int, first_image: str | None, first_video: str | None) -> dict:
    """
    The whole generation as one ComfyUI prompt, with the segments unrolled.

    Unrolled rather than one call per segment because the frames have to stay inside ComfyUI to be the next
    segment's conditioning: round-tripping them through disk between jobs would mean encoding and re-decoding
    every segment, and at 480p that is minutes of pointless VAE work per pass.

    The shape follows Kijai's LongCat_TI2V workflow exactly -- previous frames are taken with an overlap of 13,
    encoded, and handed to WanVideoEmptyEmbeds as `extra_latents`, which is what conditions the next pass.
    """
    width = int(job_input.get("width", 832))
    height = int(job_input.get("height", 480))
    if width % 16 or height % 16:
        raise InputError("width and height must both be divisible by 16")

    steps = int(job_input.get("steps") or STEPS)
    cfg = float(job_input.get("cfg") or CFG)
    shift = float(job_input.get("shift") or SHIFT)
    seed = int(job_input.get("seed", 42))
    swap = int(job_input.get("blocks_to_swap", os.environ.get("LONGCAT_BLOCK_SWAP", 0)))

    vae_tiles = {"enable_vae_tiling": False, "tile_x": 272, "tile_y": 272,
                 "tile_stride_x": 144, "tile_stride_y": 128}

    g: dict = {
        "blockswap": {"class_type": "WanVideoBlockSwap", "inputs": {
            "blocks_to_swap": swap, "offload_img_emb": False, "offload_txt_emb": False,
            "use_non_blocking": True, "vace_blocks_to_swap": 0, "prefetch_blocks": 1,
            "block_swap_debug": False}},
        "lora": {"class_type": "WanVideoLoraSelect", "inputs": {
            "lora": DISTILL_LORA, "strength": 1.0, "low_mem_load": False, "merge_loras": True}},
        "model": {"class_type": "WanVideoModelLoader", "inputs": {
            "model": DIT, "base_precision": "bf16", "quantization": QUANTIZATION,
            "load_device": "offload_device", "attention_mode": ATTENTION,
            "block_swap_args": ["blockswap", 0], "lora": ["lora", 0]}},
        "vae": {"class_type": "WanVideoVAELoader", "inputs": {
            "model_name": VAE, "precision": "bf16"}},
        "text": {"class_type": "WanVideoTextEncodeCached", "inputs": {
            "model_name": TEXT_ENCODER, "precision": "bf16",
            "positive_prompt": str(job_input.get("prompt") or ""),
            "negative_prompt": str(job_input.get("negative_prompt") or DEFAULT_NEGATIVE),
            # The file is already fp8, so asking the node to quantize again would be a second conversion.
            "quantization": "disabled", "use_disk_cache": True, "device": "gpu"}},
    }

    # What the first segment is conditioned on, if anything.
    first_conditioning = None
    if first_image:
        g["load_image"] = {"class_type": "LoadImage", "inputs": {"image": first_image}}
        g["resize"] = {"class_type": "ImageResizeKJv2", "inputs": {
            "image": ["load_image", 0], "width": width, "height": height,
            "upscale_method": "lanczos", "keep_proportion": "crop", "pad_color": "0, 0, 0",
            "crop_position": "center", "divisible_by": 16, "device": "cpu"}}
        first_conditioning = ["resize", 0]
    elif first_video:
        # Carrying on from a clip the caller supplied: its tail is the conditioning, exactly as a segment
        # boundary inside one job would be.
        g["load_video"] = {"class_type": "VHS_LoadVideoPath", "inputs": {
            "video": str(INPUT_DIR / first_video), "force_rate": FPS, "force_size": "Disabled",
            "custom_width": width, "custom_height": height, "frame_load_cap": SEGMENT_FRAMES,
            "skip_first_frames": 0, "select_every_nth": 1}}
        first_conditioning = ["load_video", 0]

    stitched = None
    for index in range(segments):
        tag = f"s{index}"
        if index == 0 and first_conditioning is None:
            embeds_inputs = {"width": width, "height": height, "num_frames": SEGMENT_FRAMES}
        else:
            source = first_conditioning if index == 0 else [f"decode_s{index - 1}", 0]
            # With no `new_images`, this hands back the overlap region to condition on rather than a join.
            g[f"cond_{tag}"] = {"class_type": "ImageBatchExtendWithOverlap", "inputs": {
                "source_images": source, "overlap": OVERLAP,
                "overlap_side": "source", "overlap_mode": "cut"}}
            g[f"enc_{tag}"] = {"class_type": "WanVideoEncode", "inputs": {
                "vae": ["vae", 0], "image": [f"cond_{tag}", 0],
                "noise_aug_strength": 0.0, "latent_strength": 1.0, **vae_tiles}}
            embeds_inputs = {"width": width, "height": height, "num_frames": SEGMENT_FRAMES,
                             "extra_latents": [f"enc_{tag}", 0]}

        g[f"embeds_{tag}"] = {"class_type": "WanVideoEmptyEmbeds", "inputs": embeds_inputs}
        g[f"sample_{tag}"] = {"class_type": "WanVideoSampler", "inputs": {
            "model": ["model", 0], "image_embeds": [f"embeds_{tag}", 0], "text_embeds": ["text", 0],
            "steps": steps, "cfg": cfg, "shift": shift,
            # Each segment gets its own seed so a long video is not 113 passes of the same noise.
            "seed": seed + index, "force_offload": True, "scheduler": SCHEDULER,
            "riflex_freq_index": 0}}
        g[f"decode_{tag}"] = {"class_type": "WanVideoDecode", "inputs": {
            "vae": ["vae", 0], "samples": [f"sample_{tag}", 0], "normalization": "default", **vae_tiles}}

        if stitched is None:
            stitched = [f"decode_{tag}", 0]
        else:
            # Joining, this time with both sides: the overlap is blended rather than cut so the seam between
            # segments is not a visible step.
            g[f"join_{tag}"] = {"class_type": "ImageBatchExtendWithOverlap", "inputs": {
                "source_images": stitched, "new_images": [f"decode_{tag}", 0],
                "overlap": OVERLAP, "overlap_side": "source", "overlap_mode": "linear_blend"}}
            stitched = [f"join_{tag}", 0]

    g["out"] = {"class_type": "VHS_VideoCombine", "inputs": {
        "images": stitched, "frame_rate": FPS, "loop_count": 0,
        "filename_prefix": "longcat", "format": "video/h264-mp4",
        "pingpong": False, "save_output": True}}
    return g


def run_graph(graph: dict, budget: float) -> dict:
    """Submit one prompt and wait for it, returning the history entry."""
    client_id = uuid.uuid4().hex
    submitted = requests.post(f"{COMFY_URL}/prompt", json={"prompt": graph, "client_id": client_id}, timeout=120)
    if submitted.status_code != 200:
        # ComfyUI reports a bad graph in detail, and that detail is the only thing that explains it.
        raise InputError(f"ComfyUI refused the graph: {submitted.text[:1200]}")
    prompt_id = submitted.json()["prompt_id"]

    deadline = time.time() + budget
    while time.time() < deadline:
        time.sleep(5)
        history = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=60).json()
        entry = history.get(prompt_id)
        if not entry:
            continue
        status = (entry.get("status") or {}).get("status_str")
        if status == "error" or (entry.get("status") or {}).get("completed") is False and status:
            messages = (entry.get("status") or {}).get("messages")
            raise RuntimeError(f"generation failed: {json.dumps(messages)[-1500:]}")
        if (entry.get("status") or {}).get("completed"):
            return entry
    raise RuntimeError(f"generation was still running after {budget / 60:.0f} minutes")


def _newest_video() -> Path:
    files = sorted(glob.glob(str(OUTPUT_DIR / "**" / "*.mp4"), recursive=True), key=os.path.getmtime)
    if not files:
        raise RuntimeError("the graph completed but produced no mp4")
    return Path(files[-1])


def _upload(path: Path, key: str):
    bucket = os.environ.get("R2_BUCKET") or os.environ.get("S3_BUCKET")
    if not bucket:
        return None

    import boto3

    account = os.environ.get("R2_ACCOUNT_ID")
    endpoint = os.environ.get("S3_ENDPOINT_URL") or (
        f"https://{account}.r2.cloudflarestorage.com" if account else None
    )
    client = boto3.client(
        "s3", endpoint_url=endpoint,
        aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("AWS_DEFAULT_REGION", "auto"),
    )
    content_type = mimetypes.guess_type(str(path))[0] or "video/mp4"
    client.upload_file(str(path), bucket, key, ExtraArgs={"ContentType": content_type})
    public_base = os.environ.get("R2_PUBLIC_BASE_URL")
    if public_base:
        return f"{public_base.rstrip('/')}/{key}"
    return client.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=7 * 24 * 3600)


def handler(job):
    job_input = job.get("input") or {}
    started = time.time()

    try:
        start_comfy()

        mode = str(job_input.get("mode", "t2v"))
        if mode not in {"t2v", "i2v", "continue"}:
            raise InputError(f"unknown mode {mode!r}; use t2v, i2v or continue")
        if mode in {"t2v", "i2v"} and not str(job_input.get("prompt") or "").strip():
            raise InputError("prompt is required for t2v and i2v")

        first_image = _stage_input(job_input, "image") if mode == "i2v" else None
        if mode == "i2v" and not first_image:
            raise InputError("i2v needs image_url or image_b64")
        first_video = _stage_input(job_input, "video") if mode == "continue" else None
        if mode == "continue":
            if not first_video:
                raise InputError("continue needs video_url or video_b64 -- pass back the previous job's video")
            if "VHS_LoadVideoPath" not in _available:
                raise InputError("this build cannot read a video input; VHS_LoadVideoPath is not installed")

        segments = segments_for(job_input)
        budget = float(job_input.get("time_budget_sec") or 2700)

        # Everything in one prompt, so the frames never leave ComfyUI between segments.
        graph = build_graph(job_input, segments, first_image, first_video)
        print(f"[longcat] {mode}, {segments} segment(s), {len(graph)} nodes", flush=True)
        run_graph(graph, budget)

        video = _newest_video()
        frames = SEGMENT_FRAMES + (segments - 1) * NEW_FRAMES_PER_SEGMENT
        response = {
            "frames": frames,
            "fps": FPS,
            "seconds": round(frames / FPS, 2),
            "segments_done": segments,
            "complete": True,
            "gpu_seconds": round(time.time() - started, 1),
            "size_bytes": video.stat().st_size,
        }
        if job_input.get("project"):
            response["project"] = job_input["project"]

        key = job_input.get("output_key") or f"longcat/{uuid.uuid4()}.mp4"
        url = _upload(video, key)
        if url:
            response["video_url"] = url
            response["output_key"] = key
            # Continuing in a later job means handing this URL back as video_url, since this worker keeps
            # nothing between jobs by design.
            response["continue_from"] = url
        elif VOLUME_DIR.is_dir():
            destination = VOLUME_DIR / key
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(video, destination)
            response["video_path"] = str(destination)
        elif video.stat().st_size <= INLINE_LIMIT_MB * 1024 * 1024:
            response["video_b64"] = base64.b64encode(video.read_bytes()).decode()
        else:
            return {"error": (
                f"result is {video.stat().st_size // (1024 * 1024)} MB and there is no bucket configured. "
                "Set R2_BUCKET and its credentials; a video of this length cannot come back inside the response."
            )}

        # The worker is reused, and a long run leaves gigabytes behind otherwise.
        video.unlink(missing_ok=True)
        for name in (first_image, first_video):
            if name:
                (INPUT_DIR / name).unlink(missing_ok=True)
        return response

    except InputError as exc:
        return {"error": str(exc)}
    except requests.RequestException as exc:
        return {"error": f"could not fetch an input: {exc}"}
    except Exception as exc:  # noqa: BLE001 - the detail is the only thing that ever explains a failure
        import traceback
        return {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-3000:]}


runpod.serverless.start({"handler": handler})
