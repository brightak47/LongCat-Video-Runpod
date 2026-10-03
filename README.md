# LongCat-Video on RunPod serverless, fp8 on a 24 GB card

[LongCat-Video](https://github.com/meituan-longcat/LongCat-Video) (Meituan, 13.6B, MIT) as a serverless worker:
text-to-video, image-to-video, and long video by continuation.

**The card it runs on is the point.** The official pipeline needs 80 GB, and on RunPod every 80 GB card is
`Low` stock in every datacentre that also supports network volumes — so jobs queue for hardware instead of
running. [Kijai's fp8 conversion](https://huggingface.co/Kijai/LongCat-Video_comfy) of the same checkpoint is
15.5 GB against 27.2 GB at bf16, which fits a 4090 alongside its activations.

`fp8_e4m3fn` is a native tensor-core format on Ada (compute 8.9), so this is not only smaller: WanVideoWrapper's
`_fast` quantization modes do the matmul in fp8. That is the difference between this and the INT8 path in the
upstream repo, which dequantizes to bf16 on every forward and so saves memory while costing speed.

**No network volume.** The ~24 GB of weights are in the image, which means no datacentre pinning, no separate
setup step, and a worker that starts wherever a 4090 is free. RunPod caches the image per machine, so the pull
is paid once per host.

The 80 GB build is kept under [`official-pipeline/`](official-pipeline/) as a fallback — it uses Meituan's own
pipeline and loads bf16 weights from a volume.

## Asking for a video

```json
{ "input": { "mode": "t2v", "prompt": "a hot air balloon rising over green hills at sunrise", "seconds": 10 } }
```

| field | meaning |
|---|---|
| `mode` | `t2v` (default), `i2v`, `continue` |
| `prompt`, `negative_prompt` | a default negative prompt is applied when none is given |
| `image_url` / `image_b64` | the first frame, for `i2v` |
| `video_url` / `video_b64` | the clip to carry on from, for `continue` |
| `seconds` or `segments` | how much video. One segment is 93 frames; each extra adds 80 more (~5.3 s) |
| `width`, `height` | default 832×480, both divisible by 16 |
| `steps`, `cfg`, `shift`, `seed` | overrides. Defaults are the distilled schedule: 10 steps, cfg 1.0, shift 12 |
| `blocks_to_swap` | transformer blocks held on CPU. 0 on a 24 GB card; raise toward 20 if memory runs out |
| `time_budget_sec` | give up after this, default 2700 |

Output carries `frames`, `fps`, `seconds`, `segments_done`, `gpu_seconds` and a `video_url` (or `video_b64`
for short results), plus `continue_from`.

## Long video

LongCat extends a clip 93 frames at a time, each pass conditioned on the previous segment's last 13 frames. A
minute is eleven passes; ten minutes is a hundred and thirteen.

Within one job the segments are **unrolled into a single ComfyUI graph**, so the frames never leave the process
— round-tripping them through disk would mean a VAE encode and decode per segment, which at 480p is minutes of
pointless work per pass.

Across jobs, continuation goes **through the bucket**, because this worker deliberately has no volume and so
nothing survives between jobs:

```json
{ "input": { "mode": "continue", "video_url": "<the previous job's video_url>", "prompt": "…", "segments": 20 } }
```

That also means a failure costs one chunk rather than the whole video. An `R2_BUCKET` is effectively required
for anything beyond a few seconds: a ten-minute video cannot be returned inside a response.

## Settings that are not guesses

The schedule and the overlap come from Kijai's own `LongCat_TI2V_example_01.json`, not from estimation:
**10 steps, cfg 1.0, shift 12.0, scheduler `longcat_distill_euler`**, overlap 13, 93 frames at 15 fps, with the
distilled LoRA loaded. The undistilled path is 50 steps at cfg 4.0 and costs roughly five times as much for no
benefit once that LoRA is in place.

Every node the handler builds a graph from is checked against the live `/object_info` when the worker boots.
These node packs move quickly, and a rename should be a clear error on startup rather than a puzzling failure
halfway through a paid job.

## Costs

Driven by GPU seconds on a 24 GB card. Reported third-party figures for this stack are 2–3 minutes per segment
on an **RTX 3060**, which implies well under a minute per segment on a 4090 — but nothing here has been measured
on our own hardware yet, and that is the first thing to do once the endpoint is up. Ask for `segments: 1` and
read `gpu_seconds`; everything else is multiplication.
