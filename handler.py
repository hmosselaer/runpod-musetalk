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
import base64, os, glob, tempfile, subprocess, shutil

# hf_transfer 403s ("no permits available") on this repo's Xet CDN bridge, which
# crash-loops the weight download. Force plain HTTP download instead.
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

APP = "/app"


def ensure_weights():
    """Download MuseTalk weights on first boot (kept out of the image to keep it
    small so RunPod provisions workers fast). If a network volume is mounted at
    /runpod-volume, weights go there and persist across cold starts; otherwise
    they download to the worker's local disk."""
    target = "/runpod-volume/models" if os.path.isdir("/runpod-volume") else os.path.join(APP, "models")
    app_models = os.path.join(APP, "models")
    # point /app/models at the target (MuseTalk's scripts expect ./models)
    if os.path.abspath(target) != os.path.abspath(app_models):
        os.makedirs(target, exist_ok=True)
        if not os.path.islink(app_models):
            if os.path.isdir(app_models):
                for name in os.listdir(app_models):
                    dst = os.path.join(target, name)
                    if not os.path.exists(dst):
                        shutil.move(os.path.join(app_models, name), dst)
                shutil.rmtree(app_models, ignore_errors=True)
            os.symlink(target, app_models)
    marker = os.path.join(target, "musetalkV15", "unet.pth")
    if os.path.exists(marker) and os.path.exists(os.path.join(target, "dwpose", "dw-ll_ucoco_384.pth")):
        print("weights present at %s" % target, flush=True)
        return
    # We download every weight ourselves rather than run MuseTalk's
    # download_weights.sh: that script upgrades huggingface_hub to >=1.0 (which
    # removes the `huggingface-cli download` command it relies on) AND uses the
    # obsolete `gdown --id` flag, so most weights silently fail to download.
    print("downloading MuseTalk weights to %s (first boot)..." % target, flush=True)
    _download_weights(target)


def _download_weights(models):
    from huggingface_hub import snapshot_download, hf_hub_download
    import urllib.request

    def hf_snap(repo, sub, patterns=None):
        dst = os.path.join(models, sub)
        os.makedirs(dst, exist_ok=True)
        print("  hf snapshot %s -> %s" % (repo, dst), flush=True)
        snapshot_download(repo_id=repo, local_dir=dst, allow_patterns=patterns,
                          local_dir_use_symlinks=False)

    # MuseTalk V1.5 UNet + config  (models/musetalkV15/unet.pth, musetalk.json)
    print("  hf snapshot TMElyralab/MuseTalk (musetalkV15) -> %s" % models, flush=True)
    snapshot_download(repo_id="TMElyralab/MuseTalk", local_dir=models,
                      allow_patterns=["musetalkV15/*"], local_dir_use_symlinks=False)
    # SD-VAE, Whisper
    hf_snap("stabilityai/sd-vae-ft-mse", "sd-vae",
            ["config.json", "diffusion_pytorch_model.bin", "diffusion_pytorch_model.safetensors"])
    hf_snap("openai/whisper-tiny", "whisper")
    # DWPose single checkpoint
    dw = os.path.join(models, "dwpose"); os.makedirs(dw, exist_ok=True)
    print("  hf file yzd-v/DWPose/dw-ll_ucoco_384.pth", flush=True)
    hf_hub_download(repo_id="yzd-v/DWPose", filename="dw-ll_ucoco_384.pth",
                    local_dir=dw, local_dir_use_symlinks=False)
    # Face-parse BiSeNet: resnet18 (pytorch.org) + 79999_iter.pth (Google Drive)
    fp = os.path.join(models, "face-parse-bisent"); os.makedirs(fp, exist_ok=True)
    r18 = os.path.join(fp, "resnet18-5c106cde.pth")
    if not os.path.exists(r18):
        print("  urllib resnet18-5c106cde.pth", flush=True)
        urllib.request.urlretrieve("https://download.pytorch.org/models/resnet18-5c106cde.pth", r18)
    fi = os.path.join(fp, "79999_iter.pth")
    if not os.path.exists(fi):
        print("  gdown 79999_iter.pth (Google Drive)", flush=True)
        import gdown
        gdown.download(id="154JgKpzCPW82qINcVieuPH3fZ2e0P812", output=fi, quiet=False)
    print("all weights downloaded.", flush=True)


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


ensure_weights()
runpod.serverless.start({"handler": handler})
