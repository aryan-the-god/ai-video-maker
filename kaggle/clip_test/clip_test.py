"""
CLIP SPEED TEST (runs on a Kaggle GPU).
Tries video models one by one, makes ONE short clip with each, and records
how long it took. Writes /kaggle/working/result.json and the clips as mp4.
A model that fails is skipped (the error is saved), so we learn what works.
"""
import gc, json, subprocess, sys, time, traceback
from pathlib import Path

OUT = Path("/kaggle/working")
PROMPT = ("Cinematic wide shot of two huge ancient armies facing each other on a dusty plain "
          "at sunrise, banners waving in the wind, golden light, dust in the air, epic film look")
result = {"gpu": "", "models": []}


def run(cmd):
    print(">>", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=False)


def save(): (OUT / "result.json").write_text(json.dumps(result, indent=2))


run("pip install -q -U 'diffusers>=0.35' transformers accelerate imageio imageio-ffmpeg ftfy sentencepiece")
import torch
from diffusers.utils import export_to_video

if torch.cuda.is_available():
    result["gpu"] = f"{torch.cuda.device_count()}x {torch.cuda.get_device_name(0)} " \
                    f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB)"
print("GPU:", result["gpu"] or "NONE", flush=True)
save()


def try_wan():
    from diffusers import WanPipeline
    pipe = WanPipeline.from_pretrained("Wan-AI/Wan2.2-TI2V-5B-Diffusers", torch_dtype=torch.float16)
    pipe.enable_model_cpu_offload()
    t = time.time()
    frames = pipe(prompt=PROMPT, height=480, width=832, num_frames=33,
                  num_inference_steps=20, guidance_scale=5.0).frames[0]
    return pipe, frames, time.time() - t, 33, 16


def try_ltx():
    from diffusers import LTXPipeline
    pipe = LTXPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    t = time.time()
    frames = pipe(prompt=PROMPT, width=704, height=480, num_frames=49,
                  num_inference_steps=30).frames[0]
    return pipe, frames, time.time() - t, 49, 24


for name, fn in [("Wan2.2-TI2V-5B (480p, 33 frames)", try_wan), ("LTX-Video 2B (480p, 49 frames)", try_ltx)]:
    entry = {"model": name}
    try:
        pipe, frames, secs, n, fps = fn()
        file = OUT / (name.split()[0].replace(".", "_") + ".mp4")
        export_to_video(frames, str(file), fps=fps)
        entry.update(ok=True, seconds=round(secs), clip_seconds=round(n / fps, 1), file=file.name)
        del pipe
    except Exception as e:
        traceback.print_exc()
        entry.update(ok=False, error=f"{type(e).__name__}: {str(e)[:300]}")
    gc.collect(); torch.cuda.empty_cache()
    result["models"].append(entry)
    save()
    print(entry, flush=True)

print("DONE", flush=True)
