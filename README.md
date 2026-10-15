# worker-wan

Wan 2.2 text-to-video and image-to-video as a Runpod serverless endpoint.

A Runpod serverless worker for **Wan 2.2 A14B** with the **LightX2V 4-step LoRAs**:
4 sampling steps instead of ~40, at 480p–720p and 16 fps. It uses ComfyUI v0.3.68,
running as a child process, with ComfyUI's official Wan 2.2 workflows
(`src/workflow.py`).

- Text-to-video: send a `prompt`.
- Image-to-video: send a `prompt` and an `image`. The video starts from that image.

## Model weights

The weights for both modes total about 69 GB, which is too large to bake into a
Runpod image (80 GB limit, 30-minute build step). The image contains no weights.
The worker downloads them from Hugging Face (`Comfy-Org/Wan_2.2_ComfyUI_Repackaged`,
pinned revision) when it starts:

| Where | When |
| --- | --- |
| `/runpod-volume/wan-models` | A network volume is attached (recommended): downloaded once, shared by all workers |
| `/ComfyUI/models/wan` | No network volume: every new worker downloads again (a few minutes) |

## Runpod setup

1. **Network volume** (recommended): Storage → New Network Volume, at least **100 GB**.
   Pick a datacenter that has the GPUs you want.
2. **Endpoint**: Serverless → New Endpoint → GitHub repo `shipdocs/worker-wan`, branch `main`, Dockerfile `Dockerfile`:
   - Network volume: the one from step 1 (this also pins the endpoint to that datacenter)
   - GPU: 48 GB or more (L40S, RTX 6000 Ada, RTX PRO 6000, A100, H100, B200)
   - Container disk: 20 GB with a network volume, **100 GB** without
   - Allowed CUDA versions: 12.8, 12.9, 13.0
   - Optional env: `WAN_MODES=t2v` or `i2v` to download and serve only one mode (half the weights)
3. Runpod builds the image on every push to `main` (the endpoint's Builds tab shows progress;
   roll back from there with ⋯ → Rollback).

The first worker start downloads the weights (69 GB for both modes, 38 GB for one). It checks
the free space first and stops with a clear message if the models won't fit. Watch the worker
logs for `Models ready`.

## Input

```json
{
  "input": {
    "prompt": "a fluffy orange cat chasing a red ball of yarn across a wooden floor, soft morning light",
    "image": "<optional: https URL, data URL or base64 of a PNG/JPG>",
    "negative_prompt": "blurry, low quality",
    "width": 832,
    "height": 480,
    "length": 81,
    "steps": 4,
    "seed": 42
  }
}
```

| Field | Default | Notes |
| --- | --- | --- |
| `prompt` | required | Longer, descriptive English prompts work best |
| `image` | none | Turns on image-to-video |
| `width`, `height` | 832×480, or the image's aspect ratio at ~480p | Multiples of 16. 1280×720 for 720p (slower) |
| `length` | 81 | Frames, must be 4n+1. 81 frames = 5 s at 16 fps |
| `steps` | 4 | The LoRAs are tuned for 4 |
| `cfg` | 1.0 | Keep at 1 with the 4-step LoRAs (the negative prompt has no effect at cfg 1) |
| `shift` | 5.0 | Sampling shift |
| `seed` | random | Returned in the output, so a result can be reproduced |

## Output

```json
{
  "status": "SUCCESS",
  "video": "<base64 MP4>",
  "mime_type": "video/mp4",
  "mode": "t2v",
  "seed": 42,
  "width": 832, "height": 480, "frames": 81, "fps": 16
}
```

Errors come back as `{"status": "ERROR", "error": "..."}` while the Runpod job shows
`COMPLETED`, so always check `output.status`.

## Testing

- `pytest tests`: input validation, workflow shape, GPU compatibility rule.
- `cd /worker && python -m src.selftest` (runs as the last image build step and in CI): starts
  ComfyUI in CPU mode with empty placeholder model files and submits both workflows.
  ComfyUI validates node names, inputs and model file names before queueing, so a
  mismatch with the pinned ComfyUI version fails the build.

## Updating ComfyUI or the models

- ComfyUI is pinned with `COMFYUI_REF` in the `Dockerfile`; Python packages are locked in
  `requirements-worker.txt` (regenerate with the command in `requirements-worker.in`).
- Model files and their Hugging Face revision are in `src/models.py`.
- The build-time selftest fails the build if the workflows no longer match ComfyUI.

## Vast.ai (ComfyUI in the browser)

`vast/provision-wan22.sh` downloads the Wan 2.2 models into Vast.ai's ComfyUI template
on startup. Copy the ComfyUI template in Vast.ai, set `PROVISIONING_SCRIPT` to
`https://raw.githubusercontent.com/shipdocs/worker-wan/main/vast/provision-wan22.sh`
and the disk to 150 GB.
It also adds two workflows to ComfyUI's Workflows list, `wan22_text_to_video_turbo` and
`wan22_image_to_video_turbo`: ComfyUI's official Wan 2.2 14B templates with
`enable_turbo_mode` (the 4-step LoRAs) switched on.
