"""
STAGE 3: makes the video clips on a Kaggle GPU (Wan2.2). Uses BOTH T4 GPUs at the same time.
The list of clips to make (JOBS_JSON) is filled in by the GitHub workflow before this is pushed.
"""
import json, os, subprocess, sys, time
from pathlib import Path

JOBS_JSON = r'''__JOBS__'''
MAX_HOURS = 10.5          # stop politely before Kaggle's 12-hour limit

WORKER_SRC = r'''
import json, os, sys, time, traceback
from pathlib import Path
import torch
from diffusers import WanPipeline
from diffusers.utils import export_to_video

idx, total_workers, max_hours = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
t0 = time.time()
jobs = json.loads(Path("/kaggle/working/jobs.json").read_text())[idx::total_workers]
out = Path("/kaggle/working/clips"); out.mkdir(exist_ok=True)
log = open(f"/kaggle/working/worker{idx}.log", "a", buffering=1)
def say(*a):
    msg = f"[w{idx}] " + " ".join(str(x) for x in a); print(msg, flush=True); log.write(msg + "\n")

# Load one worker at a time: two loads together can run out of CPU memory.
marker = Path("/kaggle/working/loaded0.flag")
if idx > 0:
    for _ in range(90):
        if marker.exists():
            break
        time.sleep(10)
say("loading Wan2.2 on", torch.cuda.get_device_name(0))
try:
    pipe = WanPipeline.from_pretrained("Wan-AI/Wan2.2-TI2V-5B-Diffusers", torch_dtype=torch.float16,
                                       low_cpu_mem_usage=True)
    pipe.enable_model_cpu_offload()
finally:
    if idx == 0:
        marker.write_text("ok")
say("model loaded")
NEG = "blurry, low quality, text, watermark, subtitles, distorted, deformed, cartoon, extra limbs"
done = failed = 0
for n, j in enumerate(jobs, 1):
    if (time.time() - t0) / 3600 > max_hours:
        say("time budget over, stopping"); break
    path = out / (j["clip"] + ".mp4")
    if path.exists():
        continue
    for attempt in (1, 2):
        try:
            g = torch.Generator("cpu").manual_seed(j["seed"])
            frames = pipe(prompt=j["prompt"], negative_prompt=NEG, height=480, width=832,
                          num_frames=33, num_inference_steps=20, guidance_scale=5.0,
                          generator=g).frames[0]
            export_to_video(frames, str(path), fps=16)
            done += 1
            say(f"{n}/{len(jobs)} ok {j['clip']}  ({(time.time()-t0)/60:.0f} min)")
            break
        except Exception as e:
            say(f"{j['clip']} attempt {attempt} failed: {type(e).__name__}: {str(e)[:200]}")
            traceback.print_exc(); torch.cuda.empty_cache()
    else:
        failed += 1
say(f"finished: {done} made, {failed} failed")
'''


def main():
    t0 = time.time()
    subprocess.run("pip install -q -U 'diffusers>=0.35' transformers accelerate imageio imageio-ffmpeg ftfy sentencepiece",
                   shell=True)
    out = Path("/kaggle/working")
    (out / "jobs.json").write_text(JOBS_JSON)
    (out / "worker.py").write_text(WORKER_SRC)
    jobs = json.loads(JOBS_JSON)
    import torch
    n_gpu = max(1, torch.cuda.device_count())
    print(f"{len(jobs)} clips to make on {n_gpu} GPU(s)", flush=True)

    procs = []
    for i in range(n_gpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i))
        procs.append(subprocess.Popen([sys.executable, str(out / "worker.py"), str(i), str(n_gpu),
                                       str(MAX_HOURS)], env=env))
    for p in procs:
        p.wait()

    made = sorted(x.name for x in (out / "clips").glob("*.mp4"))
    (out / "clips_manifest.json").write_text(json.dumps(
        {"requested": len(jobs), "made": len(made), "gpus": n_gpu,
         "minutes": round((time.time() - t0) / 60, 1), "files": made}, indent=2))
    print(f"DONE: {len(made)}/{len(jobs)} clips in {(time.time()-t0)/60:.0f} min", flush=True)


main()
