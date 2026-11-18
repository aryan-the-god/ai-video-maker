"""
QUALITY TEST on a Kaggle GPU. For 2 prompts (a wide scene and a person) it makes:
  A) normal 480p clip
  B) the 480p clip made sharper with an AI upscaler (Real-ESRGAN x2, full precision), fitted to 720p
  C) for the person only: a 480p clip with 35 steps instead of 20
and records how long each took. (Native 720p was removed: it crashed on the T4.) Results: /kaggle/working/result.json + mp4 files.
"""
import gc, json, subprocess, time, traceback
from pathlib import Path

OUT = Path("/kaggle/working")
PROMPTS = {
    "wide": ("Wide cinematic shot of a huge ancient army marching across a dusty plain at sunrise, "
             "banners waving, golden light, dust rising, slow aerial camera move"),
    "face": ("Medium close-up of a middle-aged Indian warrior with a black turban and a thick moustache, "
             "looking worried at dawn in a misty forest, soft cinematic light, slow push-in"),
}
NEG = "blurry, low quality, text, watermark, subtitles, distorted, deformed, extra limbs"
result = {"gpu": "", "items": []}


def save():
    (OUT / "result.json").write_text(json.dumps(result, indent=2))


subprocess.run("pip install -q -U 'diffusers>=0.35' transformers accelerate imageio imageio-ffmpeg ftfy "
               "sentencepiece spandrel huggingface_hub", shell=True)
import numpy as np, torch
from PIL import Image
from diffusers import WanPipeline
from diffusers.utils import export_to_video

result["gpu"] = f"{torch.cuda.device_count()}x {torch.cuda.get_device_name(0)}"
pipe = WanPipeline.from_pretrained("Wan-AI/Wan2.2-TI2V-5B-Diffusers", torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()
try:
    pipe.vae.enable_tiling()
except Exception:
    pass


def make(prompt, h, w, seed=7, steps=20):
    g = torch.Generator("cpu").manual_seed(seed)
    return pipe(prompt=prompt, negative_prompt=NEG, height=h, width=w, num_frames=33,
                num_inference_steps=steps, guidance_scale=5.0, generator=g).frames[0]


def to_uint8(frames):
    a = np.asarray(frames)
    return (np.clip(a, 0, 1) * 255).astype(np.uint8) if a.dtype != np.uint8 else a


def record(name, fn):
    entry = {"name": name}
    try:
        t = time.time(); file = fn(); entry.update(ok=True, seconds=round(time.time() - t), file=file)
    except Exception as e:
        traceback.print_exc(); entry.update(ok=False, error=f"{type(e).__name__}: {str(e)[:250]}")
    result["items"].append(entry); save(); print(entry, flush=True)
    gc.collect(); torch.cuda.empty_cache()


def upscale_fn(frames480, path):
    from huggingface_hub import hf_hub_download
    from spandrel import ModelLoader
    w = hf_hub_download("ai-forever/Real-ESRGAN", "RealESRGAN_x2.pth")
    model = ModelLoader().load_from_file(w).cuda().float().eval()
    out = []
    for fr in to_uint8(frames480):
        x = torch.from_numpy(fr).permute(2, 0, 1)[None].float().cuda() / 255
        with torch.no_grad():
            y = model(x)
        y = torch.nan_to_num(y.float()).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
        y = (y * 255).round().astype(np.uint8)
        out.append(np.asarray(Image.fromarray(y).resize((1280, 720), Image.LANCZOS)))
    export_to_video(out, str(path), fps=16)
    del model
    return path.name


for key, prompt in PROMPTS.items():
    base = {}

    def a():
        base["f"] = make(prompt, 480, 832)
        p = OUT / f"{key}_A_480p.mp4"; export_to_video(base["f"], str(p), fps=16); return p.name
    record(f"{key}: normal 480p", a)

    if "f" in base:
        record(f"{key}: 480p + AI upscale (Real-ESRGAN) to 720p", lambda: upscale_fn(base["f"], OUT / f"{key}_B_480p_upscaled.mp4"))

    if key == "face":
        def c():
            p = OUT / "face_C_480p_35steps.mp4"; export_to_video(make(prompt, 480, 832, steps=35), str(p), fps=16); return p.name
        record("face: 480p with 35 steps", c)

print("DONE", flush=True)
