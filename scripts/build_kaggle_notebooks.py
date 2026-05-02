"""Build the three Kaggle T4-session notebooks from a shared template.

Run from the repo root:

    python scripts/build_kaggle_notebooks.py

Produces:
    WBSNet_Kaggle_Session1.ipynb   (ISIC2018 A2 seed 3407)
    WBSNet_Kaggle_Session2.ipynb   (Kvasir A2/A5/A6/A7 seed 3408)
    WBSNet_Kaggle_Session3.ipynb   (Kvasir A1/A3/A4 + ClinicDB A1/A2 seed 3408)

The notebooks are intentionally near-identical; only the scope cell differs.
See kaggle-session-plan.md for the strategy behind each session.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Shared cell sources                                                         #
# --------------------------------------------------------------------------- #

PREFLIGHT_MD = """\
## 1. Pre-flight: T4 GPU check

This notebook expects **Kaggle T4 ×1**. Switch via *Settings → Accelerator*
before running. The cell below fails fast on any other GPU.
"""

PREFLIGHT_CODE = """\
!nvidia-smi -L

import torch

if not torch.cuda.is_available():
    raise RuntimeError("No CUDA GPU. Switch Settings -> Accelerator -> GPU T4 x1.")

gpu_name = torch.cuda.get_device_name(0)
print(f"Detected GPU: {gpu_name}")
if "T4" not in gpu_name:
    raise RuntimeError(
        f"Unexpected GPU '{gpu_name}'. This Kaggle notebook is tuned for T4. "
        "If you switch to P100 or V100, re-tune batch size and timing budget."
    )
"""

CLONE_MD = """\
## 2. Clone the WBSNet repo

The notebook uses `train.py` from the repo (not inline model code) so any
stability fix on `main` is picked up automatically. Make sure Internet is ON
under *Settings → Internet*.
"""

CLONE_CODE = """\
import os
from pathlib import Path

REPO_URL = "https://github.com/MrArrogant2002/WBSNet-research-paper.git"
REPO_DIR = Path("/kaggle/working/WBSNet-research-paper")

if not REPO_DIR.exists():
    !git clone --depth 1 {REPO_URL} {REPO_DIR}
else:
    print(f"Repo already at {REPO_DIR}, pulling latest")
    !cd {REPO_DIR} && git pull --ff-only

os.chdir(REPO_DIR)
print("CWD:", os.getcwd())
print("HEAD:", end=" ")
!git rev-parse --short HEAD
"""

INSTALL_CODE = """\
!pip install --quiet -r requirements-colab.txt
!pip install --quiet --no-deps -e .
!pip install --quiet google-api-python-client google-auth

import torch
print("torch:", torch.__version__,
      "| cuda:", torch.cuda.is_available(),
      "| device:", torch.cuda.get_device_name(0))

!python3 scripts/verify_repo.py
"""

DRIVE_AUTH_MD = """\
## 3. Google Drive authentication

This notebook reads three values from **Kaggle Secrets** (*Add-ons → Secrets*):

| Secret name | Source |
|---|---|
| `GDRIVE_CLIENT_ID` | OAuth client ID (Desktop app) from Google Cloud Console |
| `GDRIVE_CLIENT_SECRET` | OAuth client secret |
| `GDRIVE_REFRESH_TOKEN` | Refresh token from a one-time local OAuth flow |

See `kaggle-session-plan.md §5.1` for the one-time setup script that
generates the refresh token. Do this **once** for your account; the same
secrets work for all three notebooks.
"""

DRIVE_AUTH_CODE = """\
from kaggle_secrets import UserSecretsClient
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

DRIVE_UPLOAD_AVAILABLE = False
drive_service = None

try:
    us = UserSecretsClient()
    creds = Credentials(
        token=None,
        refresh_token=us.get_secret("GDRIVE_REFRESH_TOKEN"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=us.get_secret("GDRIVE_CLIENT_ID"),
        client_secret=us.get_secret("GDRIVE_CLIENT_SECRET"),
        scopes=["https://www.googleapis.com/auth/drive"],
    )
    drive_service = build("drive", "v3", credentials=creds, cache_discovery=False)
    about = drive_service.about().get(fields="user(emailAddress)").execute()
    DRIVE_UPLOAD_AVAILABLE = True
    print("Authenticated as:", about["user"]["emailAddress"])
except Exception as exc:
    print(f"Drive auth unavailable: {exc}")
    print("Training will continue. Archives will stay in Kaggle output for manual upload.")
"""

DRIVE_HELPERS_CODE = """\
from pathlib import Path
import random
import time

FOLDER_ID_CACHE = {}


def _escape(name: str) -> str:
    return name.replace("'", r"\\'")


def _require_drive():
    if not globals().get("DRIVE_UPLOAD_AVAILABLE", False) or drive_service is None:
        raise RuntimeError("Drive upload is unavailable; use the Kaggle output archives.")


def _sleep_for_retry(attempt: int):
    delay = min(60, 2 ** attempt) + random.random()
    time.sleep(delay)


def _execute_with_retries(request_factory, label: str, max_retries: int = 5):
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return request_factory().execute()
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries:
                break
            print(f"[drive-retry] {label}: {exc} (attempt {attempt + 1}/{max_retries})")
            _sleep_for_retry(attempt)
    raise last_exc


def ensure_folder(name: str, parent_id: str) -> str:
    _require_drive()
    cache_key = (parent_id, name)
    if cache_key in FOLDER_ID_CACHE:
        return FOLDER_ID_CACHE[cache_key]
    q = (
        f"'{parent_id}' in parents and name='{_escape(name)}' "
        "and mimeType='application/vnd.google-apps.folder' and trashed=false"
    )
    res = _execute_with_retries(
        lambda: drive_service.files().list(q=q, fields="files(id)", pageSize=1),
        f"lookup folder {name}",
    )
    if res["files"]:
        folder_id = res["files"][0]["id"]
        FOLDER_ID_CACHE[cache_key] = folder_id
        return folder_id
    body = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }
    folder_id = _execute_with_retries(
        lambda: drive_service.files().create(body=body, fields="id"),
        f"create folder {name}",
    )["id"]
    FOLDER_ID_CACHE[cache_key] = folder_id
    return folder_id


def ensure_path(parts):
    parent = "root"
    for part in parts:
        parent = ensure_folder(part, parent)
    return parent


def _run_resumable_request(request, label: str, max_retries: int = 5):
    response = None
    retries = 0
    while response is None:
        try:
            status, response = request.next_chunk()
            if status:
                print(f"[drive-upload] {label}: {status.progress() * 100:.1f}%")
            retries = 0
        except Exception as exc:
            if retries >= max_retries:
                raise
            retries += 1
            print(f"[drive-retry] {label}: {exc} (chunk retry {retries}/{max_retries})")
            _sleep_for_retry(retries)
    return response


def upload_file(local_path: Path, parent_id: str):
    _require_drive()
    q = (
        f"'{parent_id}' in parents and name='{_escape(local_path.name)}' "
        "and trashed=false"
    )
    res = _execute_with_retries(
        lambda: drive_service.files().list(q=q, fields="files(id)", pageSize=1),
        f"lookup file {local_path.name}",
    )
    media = MediaFileUpload(
        str(local_path),
        chunksize=64 * 1024 * 1024,
        resumable=True,
    )
    if res["files"]:
        request = drive_service.files().update(
            fileId=res["files"][0]["id"],
            media_body=media,
            fields="id",
        )
    else:
        request = drive_service.files().create(
            body={"name": local_path.name, "parents": [parent_id]},
            media_body=media,
            fields="id",
        )
    return _run_resumable_request(request, local_path.name)


def upload_directory(local_root: Path, parent_id: str):
    # Compatibility helper for small manual uploads. The notebook's normal path
    # uploads one archive per completed run to avoid Drive API request bursts.
    if not local_root.exists():
        return 0
    n = 0
    for entry in sorted(local_root.iterdir()):
        if entry.is_dir():
            sub_id = ensure_folder(entry.name, parent_id)
            n += upload_directory(entry, sub_id)
        else:
            try:
                upload_file(entry, parent_id)
                n += 1
            except Exception as exc:
                print(f"  ! failed to upload {entry}: {exc}")
    return n


print("Drive helpers ready. Normal flow uploads one archive file per completed run.")
"""

DATASET_MD = """\
## 4. Dataset

Preferred: attach a Kaggle Dataset whose root mirrors the
`MyDrive/WBSNet_Dataset/` layout (subdirs `kvasir/`, `cvc_clinicdb/`,
`cvc_colondb/`, `isic2018/`, each with `train/`, `val/`, `test/` and
`images/`, `masks/`, `boundaries/` underneath).

Set `KAGGLE_DATASET_PATH` below to the `/kaggle/input/<slug>/` root for
your attached dataset. The fallback path uses Drive (slow on Kaggle).
"""

DATASET_CODE = """\
KAGGLE_DATASET_PATH = Path("/kaggle/input/wbsnet-processed")

DATASET_MAP = {
    "kvasir": "Kvasir-SEG",
    "cvc_clinicdb": "CVC-ClinicDB",
    "cvc_colondb": "CVC-ColonDB",
    "isic2018": "ISIC2018",
}

DATA_ROOT = REPO_DIR / "data"
DATA_ROOT.mkdir(exist_ok=True)

if not KAGGLE_DATASET_PATH.exists():
    raise RuntimeError(
        f"Processed dataset not found at {KAGGLE_DATASET_PATH}. "
        "Attach a Kaggle Dataset (Add data button on the right) whose root "
        "contains kvasir/, cvc_clinicdb/, cvc_colondb/, isic2018/."
    )

for src_name, dst_name in DATASET_MAP.items():
    src = KAGGLE_DATASET_PATH / src_name
    dst = DATA_ROOT / dst_name
    if not src.exists():
        print(f"  skip: {src} not present")
        continue
    if dst.is_symlink():
        dst.unlink()
    elif dst.exists():
        print(f"  warn: {dst} exists and is not a symlink, leaving as-is")
        continue
    os.symlink(src, dst)
    print(f"  linked: {src} -> {dst}")

!ls -la data/
"""

SMOKE_MD = """\
## 5. Smoke test (1 epoch, batch 2)

Catches dataset path / dependency / GPU issues before the long run begins.
~3 min on T4. Skip with `RUN_SMOKE_TEST = False` if you have already
validated the setup in this session.
"""

SMOKE_CODE = """\
RUN_SMOKE_TEST = True

if RUN_SMOKE_TEST:
    !python3 train.py --config configs/kvasir_wbsnet.yaml \\
        --override train.epochs=1 train.batch_size=2 train.seed=999 \\
                   dataset.split_strategy=pre_split_dirs \\
                   dataset.num_workers=2 dataset.prefetch_factor=2 \\
                   evaluation.compute_hd95=false evaluation.max_visualizations=0 \\
                   runtime.wandb.mode=offline runtime.amp=true \\
                   experiment.name=smoke_test \\
                   experiment.run_name=smoke_test_kaggle
    print("Smoke test finished.")
else:
    print("Smoke test skipped.")
"""

UPLOADER_MD = """\
## 6. Kaggle output budget and archive upload

Kaggle output is capped at 20 GB. Instead of recursively syncing thousands of
files to Drive during training, each completed run is pruned, packed into one
`.tar.gz` archive under `/kaggle/working/wbsnet_archives/`, and then uploaded
as a single resumable Drive file. If Drive auth or quota fails, training keeps
going and the archive remains in Kaggle output.
"""

UPLOADER_CODE = """\
import shutil
import tarfile

KAGGLE_OUTPUT_ROOT = Path("/kaggle/working")
KAGGLE_OUTPUT_LIMIT_GB = 20.0
KAGGLE_OUTPUT_WARN_GB = 18.0
UPLOAD_ARCHIVE_AFTER_EACH_RUN = True
ARCHIVE_ROOT = KAGGLE_OUTPUT_ROOT / "wbsnet_archives" / DRIVE_SESSION_NAME
UPLOAD_MARKERS = ARCHIVE_ROOT / "_upload_markers"
LOCAL_PAPER_SUITE = REPO_DIR / "outputs" / "paper_suite"

ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
UPLOAD_MARKERS.mkdir(parents=True, exist_ok=True)
LOCAL_PAPER_SUITE.mkdir(parents=True, exist_ok=True)

if globals().get("DRIVE_UPLOAD_AVAILABLE", False):
    DRIVE_ARCHIVE_ROOT_ID = ensure_path([DRIVE_SESSION_NAME, "archives", "paper_suite"])
    print(f"Drive archive root ready: MyDrive/{DRIVE_SESSION_NAME}/archives/paper_suite/")
else:
    DRIVE_ARCHIVE_ROOT_ID = None
    print("Drive upload disabled for now; archives will stay in Kaggle output.")


def _iter_files(root: Path):
    if root.is_file():
        yield root
        return
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_file():
            yield path


def path_size_bytes(path: Path) -> int:
    total = 0
    for file_path in _iter_files(path):
        try:
            total += file_path.stat().st_size
        except FileNotFoundError:
            pass
    return total


def _gb(num_bytes: int) -> float:
    return num_bytes / (1024 ** 3)


def kaggle_output_used_gb() -> float:
    return _gb(path_size_bytes(KAGGLE_OUTPUT_ROOT))


def check_kaggle_output_budget(label: str = ""):
    used = kaggle_output_used_gb()
    suffix = f" after {label}" if label else ""
    print(f"[storage] /kaggle/working uses {used:.2f} GB / {KAGGLE_OUTPUT_LIMIT_GB:.1f} GB{suffix}")
    if used > KAGGLE_OUTPUT_LIMIT_GB:
        raise RuntimeError(
            f"Kaggle output exceeded {KAGGLE_OUTPUT_LIMIT_GB:.1f} GB. "
            "Delete old archives or lower visualization/checkpoint output before continuing."
        )
    if used > KAGGLE_OUTPUT_WARN_GB:
        print("[storage-warning] Close to Kaggle's 20 GB output cap.")
    return used


def archive_path_for(dataset, variant, seed, run_name):
    return (
        ARCHIVE_ROOT / "paper_suite" / dataset / variant / f"seed_{seed}"
        / f"{run_name}.tar.gz"
    )


def upload_marker_path(dataset, variant, seed):
    return UPLOAD_MARKERS / f"{dataset}_{variant}_seed{seed}.uploaded"


def archive_upload_marker_path(archive_path: Path):
    rel = archive_path.relative_to(ARCHIVE_ROOT)
    safe_name = "__".join(rel.parts).replace(".tar.gz", ".uploaded")
    return UPLOAD_MARKERS / safe_name


def run_output_dir_for(dataset, variant, seed):
    run_name = f"{dataset}_{variant}_seed{seed}"
    return (
        REPO_DIR / "outputs" / "paper_suite" / dataset / variant
        / f"seed_{seed}" / run_name
    )


def prune_completed_run(run_output_dir: Path):
    removed = []
    ckpt_dir = run_output_dir / "checkpoints"
    if ckpt_dir.exists():
        for pattern in ("epoch_*.pt", "last.pt"):
            for checkpoint in ckpt_dir.glob(pattern):
                try:
                    size_gb = _gb(checkpoint.stat().st_size)
                    checkpoint.unlink()
                    removed.append((checkpoint.name, size_gb))
                except FileNotFoundError:
                    pass
    if removed:
        freed = sum(size for _, size in removed)
        names = ", ".join(name for name, _ in removed[:5])
        more = "..." if len(removed) > 5 else ""
        print(f"[prune] removed {len(removed)} checkpoint(s), freed {freed:.2f} GB: {names}{more}")
    return removed


def make_run_archive(run_output_dir: Path, dataset, variant, seed, run_name):
    if not run_output_dir.exists():
        print(f"[archive-skip] missing run folder: {run_output_dir}")
        return None
    prune_completed_run(run_output_dir)
    archive_path = archive_path_for(dataset, variant, seed, run_name)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = archive_path.with_name(archive_path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    print(f"[archive] writing {archive_path}")
    with tarfile.open(tmp_path, "w:gz") as tar:
        tar.add(run_output_dir, arcname=run_name)
    tmp_path.replace(archive_path)
    print(f"[archive] size: {_gb(archive_path.stat().st_size):.2f} GB")
    return archive_path


def _drive_parent_for_archive(archive_path: Path):
    rel = archive_path.relative_to(ARCHIVE_ROOT)
    return ensure_path([DRIVE_SESSION_NAME, "archives", *rel.parts[:-1]])


def upload_archive_to_drive(archive_path: Path, dataset=None, variant=None, seed=None) -> bool:
    if not archive_path or not archive_path.exists():
        return False
    if not globals().get("DRIVE_UPLOAD_AVAILABLE", False):
        print(f"[drive-skip] Drive unavailable; kept local archive {archive_path}")
        return False
    marker = archive_upload_marker_path(archive_path)
    legacy_marker = None
    if dataset is not None and variant is not None and seed is not None:
        legacy_marker = upload_marker_path(dataset, variant, seed)
    if marker.exists() or (legacy_marker is not None and legacy_marker.exists()):
        print(f"[drive-skip] already uploaded: {archive_path.name}")
        return True
    try:
        parent_id = _drive_parent_for_archive(archive_path)
        upload_file(archive_path, parent_id)
        marker.write_text("uploaded\\n", encoding="utf-8")
        if legacy_marker is not None:
            legacy_marker.write_text("uploaded\\n", encoding="utf-8")
        print(f"[drive-done] uploaded archive {archive_path.name}")
        return True
    except Exception as exc:
        print(f"[drive-warning] upload failed for {archive_path.name}: {exc}")
        print("[drive-warning] Training will continue; retry from the final upload cell.")
        return False


def upload_all_archives_to_drive() -> int:
    uploaded = 0
    for archive_path in sorted(ARCHIVE_ROOT.rglob("*.tar.gz")):
        if upload_archive_to_drive(archive_path):
            uploaded += 1
    return uploaded


def finalize_run_artifacts(run_output_dir: Path, dataset, variant, seed, run_name):
    archive_path = make_run_archive(run_output_dir, dataset, variant, seed, run_name)
    if archive_path is None:
        return None
    shutil.rmtree(run_output_dir, ignore_errors=True)
    print(f"[cleanup] removed expanded run folder {run_output_dir}")
    check_kaggle_output_budget(f"archiving {run_name}")
    if UPLOAD_ARCHIVE_AFTER_EACH_RUN:
        upload_archive_to_drive(archive_path, dataset, variant, seed)
    return archive_path


check_kaggle_output_budget("setup")
print(f"Local archive root: {ARCHIVE_ROOT}")
"""

RUN_LOOP_MD = """\
## 7. Main training loop

For each `(dataset, variant, seed)` in `SCOPE`, run `train.py` with the
right config + overrides. Skips configs whose `best.pt` already exists in
`outputs/paper_suite/...` (idempotent re-runs). Skips remaining configs if
the elapsed wall-clock exceeds `SAFE_HOURS`.

Storage behavior:
- interval checkpoints are disabled (`train.save_every=0`)
- the completed run folder is pruned to keep `best.pt` and metadata
- the pruned run is archived into Kaggle output
- the expanded run folder is removed after archive creation
"""

RUN_LOOP_CODE = """\
import subprocess
import time

# Per-dataset config overrides (the Kvasir ablation configs default to
# Kvasir; we only need explicit dataset configs for ClinicDB / ISIC2018).
VARIANT_CONFIG_KVASIR = {
    "A1": "configs/ablation_identity_unet.yaml",
    "A2": "configs/kvasir_wbsnet.yaml",
    "A3": "configs/ablation_lfsa_only.yaml",
    "A4": "configs/ablation_hfba_only.yaml",
    "A5": "configs/ablation_no_boundary_supervision.yaml",
    "A6": "configs/ablation_no_wavelet.yaml",
    "A7": "configs/ablation_db2_wavelet.yaml",
}
DATASET_OVERRIDE_CONFIG = {
    ("cvc_clinicdb", "A1"): "configs/clinicdb_unet_baseline.yaml",
    ("cvc_clinicdb", "A2"): "configs/clinicdb_wbsnet.yaml",
    ("isic2018", "A1"): "configs/isic2018_unet_baseline.yaml",
    ("isic2018", "A2"): "configs/isic2018_wbsnet.yaml",
}

# Time guard: leave a 30-min buffer for final upload + cleanup.
SAFE_HOURS = 11.5
session_start = time.time()

# Archive each completed run immediately. This is a single Drive file upload,
# not a recursive per-file sync. If Drive fails, the local archive remains.
UPLOAD_ARCHIVE_AFTER_EACH_RUN = True


def remaining_hours():
    return SAFE_HOURS - (time.time() - session_start) / 3600


def run_name_for(dataset, variant, seed):
    return f"{dataset}_{variant}_seed{seed}"


def completed_archive_exists(dataset, variant, seed):
    run_name = run_name_for(dataset, variant, seed)
    return (
        archive_path_for(dataset, variant, seed, run_name).exists()
        or upload_marker_path(dataset, variant, seed).exists()
    )


def best_pt_exists(dataset, variant, seed):
    return (run_output_dir_for(dataset, variant, seed) / "checkpoints" / "best.pt").exists()


def config_for(dataset, variant):
    if (dataset, variant) in DATASET_OVERRIDE_CONFIG:
        return DATASET_OVERRIDE_CONFIG[(dataset, variant)]
    if dataset == "kvasir":
        return VARIANT_CONFIG_KVASIR[variant]
    raise KeyError(f"No config defined for ({dataset}, {variant})")


completed, skipped_done, skipped_time = [], [], []

for dataset, variant, seed in SCOPE:
    label = f"{dataset}/{variant}/seed_{seed}"
    run_name = run_name_for(dataset, variant, seed)
    if completed_archive_exists(dataset, variant, seed):
        print(f"[skip-done] {label}: archive already exists or was uploaded")
        skipped_done.append(label)
        continue
    if best_pt_exists(dataset, variant, seed):
        print(f"[archive-existing] {label}: best.pt already exists, archiving it")
        finalize_run_artifacts(run_output_dir_for(dataset, variant, seed), dataset, variant, seed, run_name)
        skipped_done.append(label)
        continue
    rem = remaining_hours()
    print(f"[start] {label}  (remaining session budget: {rem:.2f} h)")
    if rem < 1.5:
        print(f"[skip-time] {label}: only {rem:.2f} h left, deferring to Colab")
        skipped_time.append(label)
        continue
    check_kaggle_output_budget(f"before {label}")

    config = config_for(dataset, variant)
    experiment = f"paper_suite/{dataset}/{variant}/seed_{seed}"

    overrides = [
        f"train.epochs={EPOCHS_PER_RUN}",
        f"train.batch_size={BATCH_SIZE}",
        "train.grad_accum_steps=2",
        f"train.seed={seed}",
        "train.encoder_lr=0.00005",
        "train.decoder_lr=0.0005",
        "train.nonfinite_grad_action=skip",
        "train.max_nonfinite_grad_steps=10",
        "train.save_every=0",
        "dataset.split_strategy=pre_split_dirs",
        "dataset.num_workers=2",
        "dataset.prefetch_factor=2",
        "runtime.device=cuda",
        "runtime.amp=true",
        "runtime.wandb.mode=offline",
        "evaluation.compute_hd95=true",
        "evaluation.max_visualizations=4",
        f"experiment.name={experiment}",
        f"experiment.run_name={run_name}",
    ]

    cmd = ["python3", "train.py", "--config", config, "--override", *overrides]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(REPO_DIR))
    elapsed = (time.time() - t0) / 60
    if proc.returncode != 0:
        print(f"[fail] {label}: train.py exit {proc.returncode} after {elapsed:.1f} min")
        check_kaggle_output_budget(f"failed {label}")
        continue
    print(f"[done] {label}: {elapsed:.1f} min")
    completed.append(label)
    finalize_run_artifacts(run_output_dir_for(dataset, variant, seed), dataset, variant, seed, run_name)


print("\\n=== Session summary ===")
print("completed:", completed)
print("skipped (already done):", skipped_done)
print("skipped (out of time):", skipped_time)
"""

FLUSH_MD = """\
## 8. Final flush + summary

Uploads any local archives that did not make it to Drive during the run and
prints the final Kaggle output usage. Re-run this cell later if Drive quota
temporarily blocked an upload.
"""

FLUSH_CODE = """\
print("Final archive upload check...")
n_uploaded = upload_all_archives_to_drive()
print(f"Final upload pass complete: {n_uploaded} archive(s) uploaded or already present.")
check_kaggle_output_budget("final upload")

print("\\nLocal Kaggle archives:")
import subprocess as _sp
_sp.run(
    ["find", str(ARCHIVE_ROOT), "-name", "*.tar.gz", "-printf", "%p %s bytes\\n"],
    cwd=str(KAGGLE_OUTPUT_ROOT),
    check=False,
)

print("\\nExpanded best.pt files still on disk, if any:")
_sp.run(
    ["find", "outputs/paper_suite", "-name", "best.pt", "-printf", "%p %s bytes\\n"],
    cwd=str(REPO_DIR),
    check=False,
)
"""

WRAP_MD_TEMPLATE = """\
## 9. What to do next

This was **{session_label}**. Per `kaggle-session-plan.md`:

{next_steps}

If any config in this session ended up in `skipped (out of time)` above,
add it to the Colab queue manually — Colab Section 6 will pick up
everything that did finish.
"""


# --------------------------------------------------------------------------- #
# Per-notebook scope                                                          #
# --------------------------------------------------------------------------- #

SESSIONS = [
    {
        "filename": "WBSNet_Kaggle_Session1.ipynb",
        "title": "WBSNet — Kaggle Session 1 — ISIC2018 A2 seed 3407",
        "intro": dedent(
            """\
            # WBSNet — Kaggle Session 1

            **Scope:** complete seed 3407 by training the missing
            ISIC2018 A2 (full WBSNet) variant.

            **Why:** the existing legacy folder
            `MyDrive/wbsnet_paper_runs/` already contains every Kvasir and
            CVC-ClinicDB variant for seed 3407, plus ISIC2018 A1
            (identity-UNet baseline). Only ISIC2018 A2 (full WBSNet) is
            outstanding.

            **Time budget:** ~9–12 h on Kaggle T4. Single config, runs
            comfortably within the 12 h session limit.

            **Drive output:**
            `MyDrive/wbsnet_kaggle_session1/archives/paper_suite/isic2018/A2/seed_3407/*.tar.gz`

            See `kaggle-session-plan.md` for the full strategy.
            """
        ),
        "session_label": "Session 1 — ISIC2018 A2 seed 3407",
        "drive_session_name": "wbsnet_kaggle_session1",
        "scope": [("isic2018", "A2", 3407)],
        "epochs_per_run": 150,
        "batch_size": 8,
        "next_steps": (
            "- Verify Drive or Kaggle output contains the session `.tar.gz` archive.\n"
            "- In Colab Section 6, the archive is extracted before legacy import.\n"
            "- Run **Session 2** (Kvasir A2/A5/A6/A7 seed 3408)."
        ),
    },
    {
        "filename": "WBSNet_Kaggle_Session2.ipynb",
        "title": "WBSNet — Kaggle Session 2 — Kvasir A2/A5/A6/A7 seed 3408",
        "intro": dedent(
            """\
            # WBSNet — Kaggle Session 2

            **Scope:** Kvasir A2, A5, A6, A7 for seed 3408 — the four
            wavelet-relevant ablation variants.

            **Why these four:**
            - A2 = full WBSNet (haar) — paper headline row
            - A5 = full WBSNet without boundary supervision
            - A6 = LFSA + HFBA without wavelet decomposition
            - A7 = full WBSNet with db2 wavelet

            Combined with the existing seed-3407 results, this gives
            **n=2** for the most-cited ablation comparisons in the paper.

            **Time budget:** ~10–14 h on Kaggle T4. Tight — the elapsed-time
            guard will skip the last variant if it would not fit.

            **Drive output:**
            `MyDrive/wbsnet_kaggle_session2/archives/paper_suite/kvasir/{A2,A5,A6,A7}/seed_3408/*.tar.gz`

            See `kaggle-session-plan.md` for the full strategy.
            """
        ),
        "session_label": "Session 2 — Kvasir A2/A5/A6/A7 seed 3408",
        "drive_session_name": "wbsnet_kaggle_session2",
        "scope": [
            ("kvasir", "A2", 3408),
            ("kvasir", "A5", 3408),
            ("kvasir", "A6", 3408),
            ("kvasir", "A7", 3408),
        ],
        "epochs_per_run": 150,
        "batch_size": 8,
        "next_steps": (
            "- Verify each `kvasir/<variant>/seed_3408/` folder has a `.tar.gz` archive.\n"
            "- In Colab Section 6, the archives are extracted before legacy import.\n"
            "- Run **Session 3** (Kvasir A1/A3/A4 + ClinicDB A1/A2 seed 3408)."
        ),
    },
    {
        "filename": "WBSNet_Kaggle_Session3.ipynb",
        "title": "WBSNet — Kaggle Session 3 — Kvasir A1/A3/A4 + ClinicDB A1/A2 seed 3408",
        "intro": dedent(
            """\
            # WBSNet — Kaggle Session 3

            **Scope:** Kvasir A1, A3, A4 + ClinicDB A1, A2 for seed 3408.
            This closes out the seed-3408 ablation table for Kvasir and
            both ClinicDB rows.

            **Time budget:** ~11–16 h on Kaggle T4. Tighter than Session 2;
            the time guard will defer the last 1–2 configs to Colab if
            necessary.

            After this session completes, only ISIC2018 A1 / A2 seed 3408
            (and optionally seed 3409) remain for the Colab A100 run.

            **Drive output:**
            `MyDrive/wbsnet_kaggle_session3/archives/paper_suite/.../*.tar.gz`

            See `kaggle-session-plan.md` for the full strategy.
            """
        ),
        "session_label": "Session 3 — Kvasir A1/A3/A4 + ClinicDB A1/A2 seed 3408",
        "drive_session_name": "wbsnet_kaggle_session3",
        "scope": [
            ("kvasir", "A1", 3408),
            ("kvasir", "A3", 3408),
            ("kvasir", "A4", 3408),
            ("cvc_clinicdb", "A1", 3408),
            ("cvc_clinicdb", "A2", 3408),
        ],
        "epochs_per_run": 150,
        "batch_size": 8,
        "next_steps": (
            "- Verify all five run archives are present on Drive or in Kaggle output.\n"
            "- Open `WBSNet_Colab.ipynb` on Colab Pro+ A100 and run\n"
            "  `--seeds 3407 3408`. Section 6 will import this session's\n"
            "  outputs along with the other two Kaggle sessions and the\n"
            "  legacy seed-3407 folder."
        ),
    },
]


# --------------------------------------------------------------------------- #
# Notebook builder                                                            #
# --------------------------------------------------------------------------- #


def code_cell(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def md_cell(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }


def scope_cell_source(scope, drive_session_name, epochs, batch_size) -> str:
    scope_lines = [f"    ({d!r}, {v!r}, {s})," for (d, v, s) in scope]
    return (
        "# Scope = which (dataset, variant, seed) tuples this session trains.\n"
        "# Edit if you need to defer or re-run a specific config.\n"
        "SCOPE = [\n" + "\n".join(scope_lines) + "\n]\n\n"
        f"DRIVE_SESSION_NAME = {drive_session_name!r}\n"
        f"EPOCHS_PER_RUN = {epochs}\n"
        f"BATCH_SIZE = {batch_size}\n\n"
        "print('Scope:')\n"
        "for d, v, s in SCOPE:\n"
        "    print(f'  {d}/{v}/seed_{s}')\n"
        "print(f'Drive archive folder: MyDrive/{DRIVE_SESSION_NAME}/archives/paper_suite/')\n"
        "print(f'Kaggle archive folder: /kaggle/working/wbsnet_archives/{DRIVE_SESSION_NAME}/')\n"
    )


def build_notebook(session: dict) -> dict:
    cells = [
        md_cell(session["intro"]),
        md_cell(PREFLIGHT_MD),
        code_cell(PREFLIGHT_CODE),
        md_cell(CLONE_MD),
        code_cell(CLONE_CODE),
        code_cell(INSTALL_CODE),
        md_cell(DRIVE_AUTH_MD),
        code_cell(DRIVE_AUTH_CODE),
        code_cell(DRIVE_HELPERS_CODE),
        md_cell(DATASET_MD),
        code_cell(DATASET_CODE),
        md_cell("## 4b. Scope for this session"),
        code_cell(
            scope_cell_source(
                session["scope"],
                session["drive_session_name"],
                session["epochs_per_run"],
                session["batch_size"],
            )
        ),
        md_cell(SMOKE_MD),
        code_cell(SMOKE_CODE),
        md_cell(UPLOADER_MD),
        code_cell(UPLOADER_CODE),
        md_cell(RUN_LOOP_MD),
        code_cell(RUN_LOOP_CODE),
        md_cell(FLUSH_MD),
        code_cell(FLUSH_CODE),
        md_cell(
            WRAP_MD_TEMPLATE.format(
                session_label=session["session_label"],
                next_steps=session["next_steps"],
            )
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
            "kaggle": {
                "accelerator": "nvidiaTeslaT4",
                "colab": {"gpuType": "T4", "provenance": []},
                "isInternetEnabled": True,
                "language": "python",
                "sourceType": "notebook",
                "isGpuEnabled": True,
            },
            "title": session["title"],
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    for session in SESSIONS:
        nb = build_notebook(session)
        out_path = REPO_ROOT / session["filename"]
        out_path.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
        print(f"  wrote {out_path.name}  ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
