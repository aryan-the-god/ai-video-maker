"""
QUALITY TEST on a Kaggle GPU.
PART 1 (this process): makes 480p clips for 2 prompts (a wide scene and a person) and a 35-step
        version of the person, and saves them. Results are written after every clip.
PART 2 (separate process, so a crash cannot lose part 1): makes the 480p clips sharper with the
        official Real-ESRGAN x2 model and fits them to 720p.
"""
import gc, json, subprocess, sys, time, traceback
from pathlib import Path

OUT = Path("/kaggle/working")
PROMPTS = {
    "wide": ("Wide cinematic shot of a huge ancient army marching across a dusty plain at sunrise, "
             "banners waving, golden light, dust rising, slow aerial camera move"),
    "face": ("Medium close-up of a middle-aged Indian warrior with a black turban and a thick moustache, "
             "looking worried at dawn in a misty forest, soft cinematic light, slow push-in"),
}
NEG = ("blurry, low quality, text, watermark, subtitles, distorted, deformed, extra limbs, tilted horizon, "
       "dutch angle, rotating camera, camera roll, shaky camera")
result = {"gpu": "", "items": []}

UPSCALE_SRC = r'''
import json, sys, time, traceback, urllib.request
from pathlib import Path
import numpy as np, torch, imageio
from PIL import Image
from diffusers.utils import export_to_video
from spandrel import ModelLoader

OUT = Path("/kaggle/working")
URLS = ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"]
res = json.loads((OUT / "result.json").read_text())

def add(entry):
    res["items"].append(entry); (OUT / "result.json").write_text(json.dumps(res, indent=2)); print(entry, flush=True)

weights = OUT / "RealESRGAN_x2plus.pth"
try:
    urllib.request.urlretrieve(URLS[0], weights)
    model = ModelLoader().load_from_file(str(weights)).cuda().float().eval()
except Exception as e:
    add({"name": "upscaler setup", "ok": False, "error": f"{type(e).__name__}: {str(e)[:250]}"}); sys.exit(0)

for key in ("wide", "face"):
    src = OUT / f"{key}_A_480p.mp4"
    if not src.exists():
        continue
    name = f"{key}: 480p + AI upscale (official Real-ESRGAN x2) to 720p"
    try:
        t = time.time()
        frames = [np.asarray(f) for f in imageio.get_reader(str(src))]
        out = []
        for fr in frames:
            x = torch.from_numpy(fr.copy()).permute(2, 0, 1)[None].float().cuda() / 255
            with torch.no_grad():
                y = model(x)
            y = torch.nan_to_num(y.float()).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
            img = Image.fromarray((y * 255).round().astype(np.uint8)).resize((1280, 720), Image.LANCZOS)
            out.append(np.asarray(img))
        p = OUT / f"{key}_B_480p_upscaled.mp4"
        export_to_video(out, str(p), fps=16)
        add({"name": name, "ok": True, "seconds": round(time.time() - t), "file": p.name})
    except Exception as e:
        traceback.print_exc(); add({"name": name, "ok": False, "error": f"{type(e).__name__}: {str(e)[:250]}"})
'''


def save():
    (OUT / "result.json").write_text(json.dumps(result, indent=2))


subprocess.run("pip install -q -U 'diffusers>=0.35' transformers accelerate imageio imageio-ffmpeg ftfy "
               "sentencepiece spandrel", shell=True)
import numpy as np, torch
from diffusers import WanPipeline
from diffusers.utils import export_to_video

result["gpu"] = f"{torch.cuda.device_count()}x {torch.cuda.get_device_name(0)}"
pipe = WanPipeline.from_pretrained("Wan-AI/Wan2.2-TI2V-5B-Diffusers", torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()
try:
    pipe.vae.enable_tiling()
except Exception:
    pass


def make(prompt, seed=7, steps=20):
    g = torch.Generator("cpu").manual_seed(seed)
    return pipe(prompt=prompt, negative_prompt=NEG, height=480, width=832, num_frames=33,
                num_inference_steps=steps, guidance_scale=5.0, generator=g).frames[0]


def record(name, filename, steps, prompt):
    entry = {"name": name}
    try:
        t = time.time()
        frames = make(prompt, steps=steps)
        export_to_video(frames, str(OUT / filename), fps=16)
        entry.update(ok=True, seconds=round(time.time() - t), file=filename)
    except Exception as e:
        traceback.print_exc(); entry.update(ok=False, error=f"{type(e).__name__}: {str(e)[:250]}")
    result["items"].append(entry); save(); print(entry, flush=True)
    gc.collect(); torch.cuda.empty_cache()


record("wide: normal 480p", "wide_A_480p.mp4", 20, PROMPTS["wide"])
record("face: normal 480p", "face_A_480p.mp4", 20, PROMPTS["face"])
record("face: 480p with 35 steps", "face_C_480p_35steps.mp4", 35, PROMPTS["face"])

# part 2: free the video model, then upscale in a separate process
del pipe
gc.collect(); torch.cuda.empty_cache()
(OUT / "upscale.py").write_text(UPSCALE_SRC)
r = subprocess.run([sys.executable, str(OUT / "upscale.py")])
print("upscale process exit code:", r.returncode, flush=True)
if r.returncode != 0:
    res = json.loads((OUT / "result.json").read_text())
    res["items"].append({"name": "upscale process", "ok": False,
                         "error": f"stopped with exit code {r.returncode} (negative = killed by the system)"})
    (OUT / "result.json").write_text(json.dumps(res, indent=2))
print("DONE", flush=True)
