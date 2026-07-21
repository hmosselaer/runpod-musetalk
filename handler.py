# -*- coding: utf-8 -*-
"""
RunPod serverless handler for MuseTalk 1.5 lip-sync.

Input  (job["input"]):
    video_base64 : base64 mp4 (the clip whose lips to re-sync — Wan output)
    audio_base64 : base64 wav/mp3 (the voiceover to sync to)
    bbox_shift   : optional int (mouth crop vertical shift; default 0)

Output:
    { "video": "<base64 mp4>" }   # lip-synced result (motion preserved)
    or { "error": "...", "log": "..." } on failure.
"""
import runpod
import base64, os, glob, tempfile, subprocess

APP = "/app"


def _write_b64(b64, path):
    if isinstance(b64, str) and b64.startswith("data:") and "," in b64:
        b64 = b64.split(",", 1)[1]
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64))


def handler(job):
    inp = job.get("input") or {}
    video_b64 = inp.get("video_base64") or inp.get("video")
    audio_b64 = inp.get("audio_base64") or inp.get("audio") or inp.get("wav_base64")
    if not video_b64 or not audio_b64:
        return {"error": "video_base64 and audio_base64 are required"}

    work = tempfile.mkdtemp(prefix="mt_")
    vpath = os.path.join(work, "input.mp4")
    apath = os.path.join(work, "input.wav")
    result_dir = os.path.join(work, "results")
    cfg = os.path.join(work, "test.yaml")
    try:
        _write_b64(video_b64, vpath)
        _write_b64(audio_b64, apath)
    except Exception as e:
        return {"error": "could not decode inputs: %s" % e}

    bbox_shift = int(inp.get("bbox_shift", 0) or 0)
    with open(cfg, "w") as f:
        f.write("task_0:\n")
        f.write('  video_path: "%s"\n' % vpath)
        f.write('  audio_path: "%s"\n' % apath)
        f.write("  bbox_shift: %d\n" % bbox_shift)

    cmd = [
        "python", "-m", "scripts.inference",
        "--inference_config", cfg,
        "--result_dir", result_dir,
        "--unet_model_path", "models/musetalkV15/unet.pth",
        "--unet_config", "models/musetalkV15/musetalk.json",
        "--version", "v15",
    ]
    p = subprocess.run(cmd, cwd=APP, capture_output=True, text=True)
    if p.returncode != 0:
        return {"error": "inference failed", "log": (p.stderr or p.stdout or "")[-2500:]}

    outs = glob.glob(os.path.join(result_dir, "**", "*.mp4"), recursive=True)
    if not outs:
        return {"error": "no output video produced", "log": (p.stdout or "")[-2500:]}

    out = max(outs, key=os.path.getmtime)
    with open(out, "rb") as f:
        return {"video": base64.b64encode(f.read()).decode("ascii")}


runpod.serverless.start({"handler": handler})
