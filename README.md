# MuseTalk Lip-Sync — RunPod Serverless Worker

Wraps [MuseTalk 1.5](https://github.com/TMElyralab/MuseTalk) as a RunPod serverless
endpoint for **UGC Ad Foundry**. It re-syncs the lips of an existing video to a
voiceover **while preserving the video's motion** (mouth-region inpainting only).

## Contract

**Input** (`input`):
| field | type | notes |
|---|---|---|
| `video_base64` | base64 mp4 | the clip to re-sync (the Wan motion clip) |
| `audio_base64` | base64 wav/mp3 | the voiceover |
| `bbox_shift` | int (optional) | mouth crop vertical shift; default 0 |

**Output:** `{ "video": "<base64 mp4>" }` — lip-synced video (motion kept).

This matches UGC Ad Foundry's lip-sync config out of the box:
```
RUNPOD_LIPSYNC_VIDEO_FIELD=video_base64
RUNPOD_LIPSYNC_AUDIO_FIELD=audio_base64
RUNPOD_LIPSYNC_OUTPUT_FIELD=video
```

## Deploy on RunPod

1. Push this folder to a new GitHub repo.
2. RunPod → **Serverless** → **New Endpoint** → **Import Git Repository** →
   select this repo (branch `main`). RunPod builds the Dockerfile and deploys it.
   - GPU: a 24 GB card (A5000 / RTX 4090 / L4) is plenty. Workers min 0 / max 1.
3. Copy the **endpoint ID** → paste it to me → I set `RUNPOD_LIPSYNC_ENDPOINT`.

## Known build risk points (if the RunPod build fails)

MuseTalk has two notoriously fiddly steps; the error log will point at one:
1. **OpenMMLab install** — `mmcv==2.0.1 / mmdet==3.1.0 / mmpose==1.1.0` are
   version-locked against torch 2.0.1 + CUDA 11.8. If `mim install` fails, the
   fix is usually a matching `mmcv` wheel URL for cu118/torch2.0.
2. **Weight download** — the Dockerfile tries `download_weights.sh`,
   `bash download_weights.sh`, then `download_weights.py`. If MuseTalk renamed
   the script, update that line (check the repo's README "Download weights").

Send me the build log if it fails and I'll patch the Dockerfile.

## Test locally / via API
See `test_input.json` for a sample request body.
