# LongCat-Video on RunPod serverless

[LongCat-Video](https://github.com/meituan-longcat/LongCat-Video) (Meituan, 13.6B, MIT) as a serverless
worker: text-to-video, image-to-video and video continuation, including long video by continuation.

The weights are **not** in the image. The base checkpoint is 83 GB, and an image that size spends most of an
hour pulling the first time a worker lands on a new machine. They live on a network volume mounted at
`/runpod-volume` instead, which also means the model can be replaced without rebuilding anything.

## Setting it up

1. **A network volume**, 150 GB, in a datacentre that actually has 80 GB GPUs. Only some have both; as of
   writing, `CA-MTL-3` is the one with A100 80 GB PCIe *and* volume support. The volume pins the endpoint to
   its datacentre, so choosing one without the card gives an endpoint whose jobs never get a worker.

2. **Populate it once** from a pod with the volume attached:

   ```bash
   pip install 'huggingface_hub>=0.23,<1.0'
   curl -sO https://raw.githubusercontent.com/brightak47/LongCat-Video-Runpod/main/download_weights.py
   python download_weights.py            # base, ~83 GB
   python download_weights.py --avatar-1.5   # the audio-driven avatar model, ~75 GB more
   ```

   A CPU pod is enough and costs a few cents an hour; nothing here needs a GPU.

3. **Deploy from the RunPod Hub**, attach the volume, and raise the endpoint's execution timeout — the default
   is 10 minutes and a long video is not.

## Asking for a video

```json
{ "input": { "mode": "t2v", "prompt": "a hot air balloon rising over green hills at sunrise", "seconds": 10 } }
```

| field | meaning |
|---|---|
| `mode` | `t2v` (default), `i2v`, `continue` |
| `prompt`, `negative_prompt` | a default negative prompt is applied when none is given |
| `image_url` / `image_b64` | the first frame, for `i2v` |
| `video_url` / `video_b64` | the clip to extend, for `continue` |
| `seconds` or `segments` | how much video to make. One segment is 93 frames; each extra adds ~5.3 s |
| `resolution` | `480p` (default) or `720p` |
| `width`, `height` | `t2v` only, both divisible by 16 (default 832x480) |
| `quality` | `fast` (16 distilled steps, default) or `standard` (50 steps, ~3x the GPU) |
| `seed`, `steps`, `guidance_scale` | overrides, if you want them |
| `refine` | a second 720p pass that also doubles the frame rate to 30. Expensive; off by default |
| `time_budget_sec` | stop generating and bank what exists, default 2700 |

Output carries `frames`, `fps`, `seconds`, `segments_done`, `complete` and `gpu_seconds`, plus a `video_url`
(when a bucket is configured), a `video_path` on the volume, or `video_b64` for small results.

## Long video

LongCat makes long video by continuation: 93 frames at a time, each segment conditioned on the tail of the
last. A minute is eleven segments and half an hour is over three hundred — more than one serverless job may
run for.

So the worker generates until `time_budget_sec` is spent, writes the tail of what it has to the volume, and
returns `"complete": false` with a `resume_key`. Pass that key back and it continues from exactly there:

```json
{ "input": { "mode": "continue", "resume_key": "a1b2c3…", "prompt": "…", "segments": 40 } }
```

The caller stitches the pieces. This is the only way a half-hour video is possible at all, and it means a
failure costs one chunk rather than the whole thing.

## Costs

Driven almost entirely by GPU seconds on an 80 GB card. `quality: "fast"` uses the distilled 16-step schedule
and is roughly a third of the standard 50-step cost for most content, so it is the default; `refine` is
another full 50-step pass over every frame and should be asked for deliberately.
