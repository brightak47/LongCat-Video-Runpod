"""RunPod serverless worker for LongCat-Video (text-to-video, image-to-video, continuation).

Input (job["input"]):
    mode            "t2v" (default) | "i2v" | "continue"
    prompt          required for t2v/i2v; optional for continue
    negative_prompt optional, a sensible default is applied
    image_url | image_b64       the first frame, for i2v
    video_url | video_b64       the clip to continue, for continue
    resume_key      continue a job this worker produced earlier (see "Long video" below)

    segments        extra continuation segments to append, default 0. Each adds ~5.3 s of video.
    seconds         asked-for length in seconds; converted to segments, and easier to reason about
    resolution      "480p" (default) or "720p"
    width, height   t2v only, both divisible by 16; defaults 832x480
    quality         "fast" (default, 16 distilled steps) or "standard" (50 steps)
    steps           overrides the step count that `quality` implies
    guidance_scale  defaults to 1.0 for fast, 4.0 for standard
    seed            defaults to 42
    refine          false by default. A second 720p pass that also doubles the frame rate to 30.
    time_budget_sec how long to keep generating before banking what exists, default 2700

    output_key      destination key in the bucket
    project         echoed back for cost attribution

Output:
    {"video_url": ...} | {"video_path": ...} | {"video_b64": ...}
    plus {"frames", "fps", "seconds", "segments_done", "complete", "resume_key"?, "project"?}

Long video:
    LongCat makes long video by continuation: 93 frames at a time, each segment conditioned on the tail of the
    last. A minute is eleven segments, and half an hour is well over three hundred -- far more than a serverless
    job may run for. So this worker generates until `time_budget_sec` is spent, writes the tail of what it has to
    the network volume, and returns `resume_key` with "complete": false. Calling it again with that key continues
    from exactly there. The caller stitches, or asks for the segments it wants and stitches once at the end.

Environment:
    CHECKPOINT_DIR   default /runpod-volume/weights/LongCat-Video
    R2_ACCOUNT_ID / R2_BUCKET / R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY   -> upload and return a URL
    S3_ENDPOINT_URL / S3_BUCKET / AWS_* also work for any S3-compatible store
"""

import base64
import mimetypes
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np
import requests
import runpod

CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/runpod-volume/weights/LongCat-Video")
VOLUME_DIR = Path("/runpod-volume")
STATE_DIR = VOLUME_DIR / "state"
INLINE_LIMIT_MB = int(os.environ.get("INLINE_LIMIT_MB", "18"))
DOWNLOAD_TIMEOUT = int(os.environ.get("DOWNLOAD_TIMEOUT", "180"))

# The model's native frame rate, and the chunk it works in. Both are properties of the checkpoint rather than
# choices: num_frames must satisfy (n - 1) % 4 == 0 for the VAE's temporal scale factor, and 93 is what every
# demo and the conditioning window are built around.
FPS = 15
SEGMENT_FRAMES = 93
COND_FRAMES = 13
# Each segment after the first contributes this many new frames, the rest being the overlap it was conditioned on.
NEW_FRAMES_PER_SEGMENT = SEGMENT_FRAMES - COND_FRAMES

DEFAULT_NEGATIVE = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, "
    "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still "
    "picture, messy background, three legs, many people in the background, walking backwards"
)


class InputError(Exception):
    """Something wrong with the request rather than the worker."""


_pipe = None


def pipeline():
    """
    Load the model once per worker.

    Eighty-three gigabytes off a network volume is minutes of cold start, so it is held for the life of the
    worker and every later job on it is warm. Imports are inside the function because importing torch at module
    scope would make even a malformed request pay for it.
    """
    global _pipe
    if _pipe is not None:
        return _pipe

    import torch
    import torch.distributed as dist
    from transformers import AutoTokenizer, UMT5EncoderModel
    from longcat_video.pipeline_longcat_video import LongCatVideoPipeline
    from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
    from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
    from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
    from longcat_video.context_parallel import context_parallel_util
    from longcat_video.context_parallel.context_parallel_util import init_context_parallel

    if not Path(CHECKPOINT_DIR).is_dir():
        raise InputError(
            f"no checkpoint at {CHECKPOINT_DIR}. Attach the weights volume to this endpoint and populate it "
            "with download_weights.py."
        )

    # The project's scripts are meant to be launched with torchrun, which sets all of this. A serverless worker
    # is one process on one GPU, so the single-process equivalent is set here rather than requiring a launcher.
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    torch.cuda.set_device(0)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    # Context parallelism splits attention across GPUs. With one GPU it is a no-op, but the DiT still asks for
    # the split it was built with, so it has to be initialised rather than skipped.
    init_context_parallel(context_parallel_size=1, global_rank=dist.get_rank(), world_size=dist.get_world_size())
    cp_split_hw = context_parallel_util.get_optimal_split(context_parallel_util.get_cp_size())

    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR, subfolder="tokenizer", torch_dtype=torch.bfloat16)
    text_encoder = UMT5EncoderModel.from_pretrained(CHECKPOINT_DIR, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    vae = AutoencoderKLWan.from_pretrained(CHECKPOINT_DIR, subfolder="vae", torch_dtype=torch.bfloat16)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(CHECKPOINT_DIR, subfolder="scheduler", torch_dtype=torch.bfloat16)
    dit = LongCatVideoTransformer3DModel.from_pretrained(CHECKPOINT_DIR, subfolder="dit", cp_split_hw=cp_split_hw, torch_dtype=torch.bfloat16)

    pipe = LongCatVideoPipeline(tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, scheduler=scheduler, dit=dit)
    pipe.to(0)
    print(f"[longcat] model loaded in {time.time() - started:.0f}s from {CHECKPOINT_DIR}", flush=True)
    _pipe = pipe
    return _pipe


def _fetch(url: str, dest: Path) -> Path:
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
        response.raise_for_status()
        with open(dest, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                handle.write(chunk)
    if dest.stat().st_size == 0:
        raise InputError(f"downloaded file is empty: {url}")
    return dest


def _resolve_media(job_input: dict, kind: str, work: Path):
    suffix = {"image": ".png", "video": ".mp4"}[kind]
    dest = work / f"input_{kind}{suffix}"
    if job_input.get(f"{kind}_url"):
        return _fetch(job_input[f"{kind}_url"], dest)
    if job_input.get(f"{kind}_b64"):
        dest.write_bytes(base64.b64decode(job_input[f"{kind}_b64"]))
        return dest
    return None


def _frames_of(array) -> list:
    """A pipeline result (float 0..1) as the PIL frames every continuation call wants."""
    import PIL.Image

    return [PIL.Image.fromarray((array[i] * 255).astype(np.uint8)) for i in range(array.shape[0])]


def _read_video(path: Path) -> list:
    import imageio.v3 as iio
    import PIL.Image

    return [PIL.Image.fromarray(frame) for frame in iio.imread(path, plugin="pyav")]


def _write_video(frames: list, path: Path, fps: int = FPS, crf: int = 18) -> Path:
    import torch
    from torchvision.io import write_video

    tensor = torch.from_numpy(np.array([np.array(f) for f in frames])).to(torch.uint8)
    write_video(str(path), tensor, fps=fps, video_codec="libx264", options={"crf": str(crf)})
    return path


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
        "s3",
        endpoint_url=endpoint,
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


def segments_for(job_input: dict) -> int:
    """
    How many continuation segments this request is asking for.

    `seconds` is what a caller actually thinks in, so it is converted here rather than making every caller do the
    arithmetic. The first 93 frames are free of charge in segment terms, which is why the initial chunk is
    subtracted before dividing.
    """
    if job_input.get("segments") is not None:
        return max(0, int(job_input["segments"]))
    seconds = job_input.get("seconds")
    if not seconds:
        return 0
    wanted_frames = float(seconds) * FPS
    return max(0, int(np.ceil((wanted_frames - SEGMENT_FRAMES) / NEW_FRAMES_PER_SEGMENT)))


def handler(job):
    job_input = job.get("input") or {}
    started = time.time()
    work = Path(tempfile.mkdtemp(prefix="longcat_", dir="/tmp"))

    try:
        import torch

        mode = str(job_input.get("mode", "t2v"))
        if mode not in {"t2v", "i2v", "continue"}:
            raise InputError(f"unknown mode {mode!r}; use t2v, i2v or continue")

        prompt = str(job_input.get("prompt") or "").strip()
        if mode in {"t2v", "i2v"} and not prompt:
            raise InputError("prompt is required for t2v and i2v")
        negative = str(job_input.get("negative_prompt") or DEFAULT_NEGATIVE)

        fast = str(job_input.get("quality", "fast")) == "fast"
        steps = int(job_input.get("steps") or (16 if fast else 50))
        guidance = float(job_input.get("guidance_scale") or (1.0 if fast else 4.0))
        resolution = str(job_input.get("resolution", "480p"))
        if resolution not in {"480p", "720p"}:
            raise InputError("resolution must be 480p or 720p")
        wanted_segments = segments_for(job_input)
        budget = float(job_input.get("time_budget_sec") or 2700)

        pipe = pipeline()
        generator = torch.Generator(device=0)
        generator.manual_seed(int(job_input.get("seed", 42)))

        common = dict(num_inference_steps=steps, use_distill=fast, guidance_scale=guidance, generator=generator)

        # The opening chunk, or the tail of an earlier job to carry on from.
        resume_key = job_input.get("resume_key")
        if resume_key:
            state = STATE_DIR / f"{Path(str(resume_key)).name}.mp4"
            if not state.is_file():
                raise InputError(f"resume_key {resume_key!r} has expired or never existed")
            frames = _read_video(state)
            produced = list(frames)
        elif mode == "t2v":
            width = int(job_input.get("width", 832))
            height = int(job_input.get("height", 480))
            if width % 16 or height % 16:
                raise InputError("width and height must both be divisible by 16")
            output = pipe.generate_t2v(prompt=prompt, negative_prompt=negative, height=height, width=width,
                                      num_frames=SEGMENT_FRAMES, **common)[0]
            frames = _frames_of(output)
            produced = list(frames)
        elif mode == "i2v":
            from diffusers.utils import load_image

            path = _resolve_media(job_input, "image", work)
            if path is None:
                raise InputError("i2v needs image_url or image_b64")
            image = load_image(str(path))
            output = pipe.generate_i2v(image=image, prompt=prompt, negative_prompt=negative, resolution=resolution,
                                       num_frames=SEGMENT_FRAMES, **common)[0]
            frames = _frames_of(output)
            produced = list(frames)
        else:
            path = _resolve_media(job_input, "video", work)
            if path is None:
                raise InputError("continue needs video_url, video_b64 or resume_key")
            frames = _read_video(path)
            if len(frames) < COND_FRAMES:
                raise InputError(f"the clip to continue has {len(frames)} frames; at least {COND_FRAMES} are needed")
            # The input is conditioning, not output: what this job produced starts with the first new segment.
            produced = []
            wanted_segments = max(1, wanted_segments)

        # Each segment is conditioned on the tail of the one before, and only its new frames are kept. The size
        # the first chunk came out at is the size everything is held to: a continuation can come back a few
        # pixels different, and frames of two sizes cannot be encoded as one video.
        target_size = frames[0].size
        done = 0
        for index in range(wanted_segments):
            spent = time.time() - started
            if spent > budget:
                print(f"[longcat] banking {done} of {wanted_segments} segments after {spent:.0f}s", flush=True)
                break
            output = pipe.generate_vc(video=frames, prompt=prompt or "", negative_prompt=negative,
                                      resolution=resolution, num_frames=SEGMENT_FRAMES,
                                      num_cond_frames=COND_FRAMES, use_kv_cache=True, offload_kv_cache=False,
                                      enhance_hf=True, **common)[0]
            segment = [f.resize(target_size) for f in _frames_of(output)]
            produced.extend(segment[COND_FRAMES:])
            frames = segment
            done += 1
            print(f"[longcat] segment {done}/{wanted_segments} at {time.time() - started:.0f}s", flush=True)

        if not produced:
            raise InputError("nothing was generated; ask for at least one segment")

        fps = FPS
        if job_input.get("refine"):
            # A second pass at 720p that also interpolates to 30 fps. Expensive -- it is another 50-step diffusion
            # over every frame -- so it is off unless asked for, and it runs over what exists rather than
            # re-generating anything.
            lora = Path(CHECKPOINT_DIR) / "lora" / "refinement_lora.safetensors"
            if not lora.is_file():
                raise InputError(f"refine was asked for but {lora} is not on the volume")
            pipe.dit.load_lora(str(lora), "refinement_lora")
            pipe.dit.enable_loras(["refinement_lora"])
            pipe.dit.enable_bsa()
            refined = pipe.generate_refine(video=None, prompt="", stage1_video=produced[:SEGMENT_FRAMES],
                                           num_cond_frames=0, num_inference_steps=50, generator=generator,
                                           spatial_refine_only=False)[0]
            produced = _frames_of(refined)
            fps = 30

        out = _write_video(produced, work / "out.mp4", fps=fps)
        complete = done >= wanted_segments

        response = {
            "frames": len(produced),
            "fps": fps,
            "seconds": round(len(produced) / fps, 2),
            "segments_done": done,
            "complete": complete,
            "gpu_seconds": round(time.time() - started, 1),
            "size_bytes": out.stat().st_size,
        }
        if job_input.get("project"):
            response["project"] = job_input["project"]

        # Unfinished work leaves its tail where the next job can pick it up. Only on the volume: there is nowhere
        # else a later worker could read it from, so without one a long video has to be asked for in one go.
        if not complete and STATE_DIR.parent.is_dir():
            token = uuid.uuid4().hex
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            _write_video(frames, STATE_DIR / f"{token}.mp4", fps=fps, crf=10)
            response["resume_key"] = token

        key = job_input.get("output_key") or f"longcat/{uuid.uuid4()}.mp4"
        url = _upload(out, key)
        if url:
            response["video_url"] = url
            response["output_key"] = key
        elif VOLUME_DIR.is_dir():
            destination = VOLUME_DIR / key
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(out, destination)
            response["video_path"] = str(destination)
        elif out.stat().st_size <= INLINE_LIMIT_MB * 1024 * 1024:
            response["video_b64"] = base64.b64encode(out.read_bytes()).decode()
        else:
            return {
                "error": (
                    f"result is {out.stat().st_size // (1024 * 1024)} MB with no bucket or volume configured; "
                    "set R2_BUCKET or attach a network volume"
                )
            }
        return response

    except InputError as exc:
        return {"error": str(exc)}
    except requests.RequestException as exc:
        return {"error": f"could not fetch an input: {exc}"}
    except Exception as exc:  # noqa: BLE001 - the tail of the traceback is the only thing that explains a failure
        import traceback

        return {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-3000:]}
    finally:
        shutil.rmtree(work, ignore_errors=True)


runpod.serverless.start({"handler": handler})
