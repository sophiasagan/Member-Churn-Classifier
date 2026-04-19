import time
from pathlib import Path

import openai

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

TRAIN_PATH = Path("data/finetune_train.jsonl")
VAL_PATH   = Path("data/finetune_val.jsonl")
MODEL_OUT  = Path("data/fine_tuned_model.txt")

BASE_MODEL = "gpt-4o-mini-2024-07-18"
SUFFIX     = "cu-churn-narrative"
N_EPOCHS   = 3
POLL_SEC   = 30

client = openai.OpenAI()

# ── upload files ──────────────────────────────────────────────────────────────
def upload(path: Path, purpose: str = "fine-tune") -> str:
    print(f"Uploading {path.name} ...", end=" ", flush=True)
    with open(path, "rb") as f:
        resp = client.files.create(file=f, purpose=purpose)
    print(f"done  (id={resp.id})")
    return resp.id

print("-- Upload ----------------------------------------------------------")
train_file_id = upload(TRAIN_PATH)
val_file_id   = upload(VAL_PATH)

# ── launch fine-tune job ──────────────────────────────────────────────────────
print("\n-- Launch fine-tune job --------------------------------------------")
job = client.fine_tuning.jobs.create(
    model=BASE_MODEL,
    training_file=train_file_id,
    validation_file=val_file_id,
    hyperparameters={"n_epochs": N_EPOCHS},
    suffix=SUFFIX,
)
print(f"Job ID   : {job.id}")
print(f"Model    : {job.model}")
print(f"Status   : {job.status}")

# ── poll until terminal ───────────────────────────────────────────────────────
print(f"\n-- Polling every {POLL_SEC}s -----------------------------------------")
terminal = {"succeeded", "failed", "cancelled"}

while job.status not in terminal:
    time.sleep(POLL_SEC)
    job = client.fine_tuning.jobs.retrieve(job.id)

    # show the most recent event if available
    events = client.fine_tuning.jobs.list_events(job.id, limit=1).data
    last   = events[0].message if events else ""
    print(f"[{job.status:>12}]  {last}")

# ── result ────────────────────────────────────────────────────────────────────
print(f"\n-- Result ----------------------------------------------------------")
print(f"Final status : {job.status}")

if job.status == "succeeded":
    model_id = job.fine_tuned_model
    print(f"Fine-tuned model : {model_id}")
    MODEL_OUT.write_text(model_id)
    print(f"Saved to {MODEL_OUT}")
else:
    # surface the failure reason from events
    events = client.fine_tuning.jobs.list_events(job.id, limit=10).data
    print("Job did not succeed. Recent events:")
    for ev in reversed(events):
        print(f"  [{ev.level}] {ev.message}")
    raise SystemExit(1)
