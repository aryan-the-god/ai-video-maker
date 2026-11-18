"""
STAGE 3: makes the video clips on a Kaggle GPU (Wan2.2). Uses BOTH T4 GPUs at the same time.
The list of clips to make (JOBS_JSON) is filled in by the GitHub workflow before this is pushed.

Memory plan (the first version ran out of CPU memory because each GPU worker kept the big
text model in memory too):
  step A: one process turns all prompts into numbers (text model on the GPU), saves them, exits
  step B: two workers (one per GPU) load ONLY the video model and use the saved numbers
"""
import json, os, subprocess, sys, time
from pathlib import Path

JOBS_JSON = r'''__JOBS__'''
MAX_HOURS = 10.5          # stop politely before Kaggle's 12-hour limit
MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
NEG = "blurry, low quality, text, watermark, subtitles, distorted, deformed, cartoon, extra limbs"

ENCODE_SRC = r'''
import json, sys, torch
from pathlib import Path
from diffusers import WanPipeline
MODEL, NEG = sys.argv[1], sys.argv[2]
jobs = json.loads(Path("/kaggle/working/jobs.json").read_text())
pipe = WanPipeline.from_pretrained(MODEL, transformer=None, vae=None, torch_dtype=torch.float16)
pipe.to("cuda")
prompts = sorted({j["prompt"] for j in jobs})
print(f"encoding {len(prompts)} different prompts", flush=True)
store = {}
for k, p in enumerate(prompts, 1):
    with torch.no_grad():
        pe, ne = pipe.encode_prompt(prompt=p, negative_prompt=NEG, do_classifier_free_guidance=True,
                                    device="cuda")
    store[p] = pe.to("cpu", torch.float16)
    store["__NEG__"] = ne.to("cpu", torch.float16)
torch.save(store, "/kaggle/working/embeds.pt")
print("embeddings saved", flush=True)
'''

WORKER_SRC = r'''
import json, os, sys, time, traceback
from pathlib import Path
import torch
from diffusers import WanPipeline
from diffusers.utils import export_to_video

MODEL = sys.argv[4]
idx, total_workers, max_hours = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
t0 = time.time()
jobs = json.loads(Path("/kaggle/working/jobs.json").read_text())[idx::total_workers]
out = Path("/kaggle/working/clips"); out.mkdir(exist_ok=True)
log = open(f"/kaggle/working/worker{idx}.log", "a", buffering=1)

def ram_free():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable"):
                return f"{int(line.split()[1]) / 1e6:.0f}GB free RAM"
    except Exception:
        pass
    return ""

def say(*a):
    msg = f"[w{idx}] " + " ".join(str(x) for x in a) + f"  [{ram_free()}]"
    print(msg, flush=True); log.write(msg + "\n")

# Load one worker at a time, so both do not use CPU memory at the same moment.
marker = Path("/kaggle/working/loaded0.flag")
if idx > 0:
    for _ in range(90):
        if marker.exists():
            break
        time.sleep(10)
say("loading video model on", torch.cuda.get_device_name(0))
try:
    pipe = WanPipeline.from_pretrained(MODEL, text_encoder=None, tokenizer=None,
                                       torch_dtype=torch.float16, low_cpu_mem_usage=True)
    pipe.enable_model_cpu_offload()
finally:
    if idx == 0:
        marker.write_text("ok")
say("model loaded")
emb = torch.load("/kaggle/working/embeds.pt")
neg = emb["__NEG__"].to("cuda")

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
            frames = pipe(prompt_embeds=emb[j["prompt"]].to("cuda"), negative_prompt_embeds=neg,
                          height=480, width=832, num_frames=33, num_inference_steps=20,
                          guidance_scale=5.0, generator=g).frames[0]
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
    (out / "encode.py").write_text(ENCODE_SRC)
    jobs = json.loads(JOBS_JSON)
    import torch
    n_gpu = max(1, torch.cuda.device_count())
    notes = [f"{len(jobs)} clips to make on {n_gpu} GPU(s)"]
    print(notes[0], flush=True)

    # step A: prompts -> numbers (separate process, so its memory is freed completely afterwards)
    r = subprocess.run([sys.executable, str(out / "encode.py"), MODEL, NEG])
    notes.append(f"encode step exit code: {r.returncode}")

    # step B: one worker per GPU
    procs = []
    if r.returncode == 0:
        for i in range(n_gpu):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i))
            procs.append(subprocess.Popen([sys.executable, str(out / "worker.py"), str(i), str(n_gpu),
                                           str(MAX_HOURS), MODEL], env=env))
        for i, p in enumerate(procs):
            notes.append(f"worker {i} exit code: {p.wait()}  (negative = killed by the system)")
    (out / "launcher.log").write_text("\n".join(notes))

    made = sorted(x.name for x in (out / "clips").glob("*.mp4")) if (out / "clips").exists() else []
    (out / "clips_manifest.json").write_text(json.dumps(
        {"requested": len(jobs), "made": len(made), "gpus": n_gpu,
         "minutes": round((time.time() - t0) / 60, 1), "files": made, "notes": notes}, indent=2))
    print(f"DONE: {len(made)}/{len(jobs)} clips in {(time.time()-t0)/60:.0f} min", flush=True)


main()
