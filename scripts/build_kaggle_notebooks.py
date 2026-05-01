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
print("Authenticated as:", about["user"]["emailAddress"])
"""

DRIVE_HELPERS_CODE = """\
from pathlib import Path
import time


def _escape(name: str) -> str:
    return name.replace("'", r"\\'")


def ensure_folder(name: str, parent_id: str) -> str:
    q = (
        f"'{parent_id}' in parents and name='{_escape(name)}' "
        "and mimeType='application/vnd.google-apps.folder' and trashed=false"
    )
    res = drive_service.files().list(q=q, fields="files(id)", pageSize=1).execute()
    if res["files"]:
        return res["files"][0]["id"]
    body = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }
    return drive_service.files().create(body=body, fields="id").execute()["id"]


def ensure_path(parts):
    parent = "root"
    for part in parts:
        parent = ensure_folder(part, parent)
    return parent


def upload_file(local_path: Path, parent_id: str):
    q = (
        f"'{parent_id}' in parents and name='{_escape(local_path.name)}' "
        "and trashed=false"
    )
    res = drive_service.files().list(q=q, fields="files(id)", pageSize=1).execute()
    resumable = local_path.stat().st_size > 5 * 1024 * 1024
    media = MediaFileUpload(str(local_path), resumable=resumable)
    if res["files"]:
        drive_service.files().update(
            fileId=res["files"][0]["id"], media_body=media
        ).execute()
    else:
        drive_service.files().create(
            body={"name": local_path.name, "parents": [parent_id]},
            media_body=media,
            fields="id",
        ).execute()


def upload_directory(local_root: Path, parent_id: str):
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


print("Drive helpers ready.")
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
## 6. Background Drive sync

Mirrors `outputs/paper_suite/` to the session's Drive root every 10 min so
nothing is lost if the Kaggle session is killed mid-run.
"""

UPLOADER_CODE = """\
import threading

class DriveSyncer:
    def __init__(self, local_root: Path, drive_root_id: str, interval_sec: int = 600):
        self.local_root = local_root
        self.drive_root_id = drive_root_id
        self.interval = interval_sec
        self._stop = threading.Event()
        self._thread = None

    def _loop(self):
        while not self._stop.is_set():
            try:
                if self.local_root.exists():
                    n = upload_directory(self.local_root, self.drive_root_id)
                    print(f"[DriveSyncer] uploaded {n} file(s)")
            except Exception as exc:
                print(f"[DriveSyncer] error: {exc}")
            self._stop.wait(self.interval)

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=60)


SESSION_DRIVE_FOLDER = ensure_folder(DRIVE_SESSION_NAME, "root")
PAPER_SUITE_DRIVE_ID = ensure_folder("paper_suite", SESSION_DRIVE_FOLDER)
LOCAL_PAPER_SUITE = REPO_DIR / "outputs" / "paper_suite"
LOCAL_PAPER_SUITE.mkdir(parents=True, exist_ok=True)

syncer = DriveSyncer(LOCAL_PAPER_SUITE, PAPER_SUITE_DRIVE_ID, interval_sec=600)
syncer.start()
print(f"Periodic Drive sync running. Target: MyDrive/{DRIVE_SESSION_NAME}/paper_suite/")
"""

RUN_LOOP_MD = """\
## 7. Main training loop

For each `(dataset, variant, seed)` in `SCOPE`, run `train.py` with the
right config + overrides. Skips configs whose `best.pt` already exists in
`outputs/paper_suite/...` (idempotent re-runs). Skips remaining configs if
the elapsed wall-clock exceeds `SAFE_HOURS`.
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


def remaining_hours():
    return SAFE_HOURS - (time.time() - session_start) / 3600


def best_pt_exists(dataset, variant, seed):
    return (
        REPO_DIR / "outputs" / "paper_suite" / dataset / variant
        / f"seed_{seed}" / f"{dataset}_{variant}_seed{seed}"
        / "checkpoints" / "best.pt"
    ).exists()


def config_for(dataset, variant):
    if (dataset, variant) in DATASET_OVERRIDE_CONFIG:
        return DATASET_OVERRIDE_CONFIG[(dataset, variant)]
    if dataset == "kvasir":
        return VARIANT_CONFIG_KVASIR[variant]
    raise KeyError(f"No config defined for ({dataset}, {variant})")


completed, skipped_done, skipped_time = [], [], []

for dataset, variant, seed in SCOPE:
    label = f"{dataset}/{variant}/seed_{seed}"
    if best_pt_exists(dataset, variant, seed):
        print(f"[skip-done] {label}: best.pt already on disk")
        skipped_done.append(label)
        continue
    rem = remaining_hours()
    print(f"[start] {label}  (remaining session budget: {rem:.2f} h)")
    if rem < 1.5:
        print(f"[skip-time] {label}: only {rem:.2f} h left, deferring to Colab")
        skipped_time.append(label)
        continue

    config = config_for(dataset, variant)
    run_name = f"{dataset}_{variant}_seed{seed}"
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
        "train.save_every=10",
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
        continue
    print(f"[done] {label}: {elapsed:.1f} min")
    completed.append(label)


print("\\n=== Session summary ===")
print("completed:", completed)
print("skipped (already done):", skipped_done)
print("skipped (out of time):", skipped_time)
"""

FLUSH_MD = """\
## 8. Final flush + summary

Stops the periodic syncer, then does one last full upload to make sure
everything (including any artifacts written between the last sync and the
end of training) is on Drive.
"""

FLUSH_CODE = """\
syncer.stop()
print("Periodic syncer stopped. Final upload starting...")

n_uploaded = upload_directory(LOCAL_PAPER_SUITE, PAPER_SUITE_DRIVE_ID)
print(f"Final upload complete: {n_uploaded} file(s).")

print("\\nLocal artifact tree:")
import subprocess as _sp
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
            `MyDrive/wbsnet_kaggle_session1/paper_suite/isic2018/A2/seed_3407/`

            See `kaggle-session-plan.md` for the full strategy.
            """
        ),
        "session_label": "Session 1 — ISIC2018 A2 seed 3407",
        "drive_session_name": "wbsnet_kaggle_session1",
        "scope": [("isic2018", "A2", 3407)],
        "epochs_per_run": 150,
        "batch_size": 8,
        "next_steps": (
            "- Verify the Drive folder contains `checkpoints/best.pt`,\n"
            "  `metrics.csv`, `run_summary.json`, `best_metrics.json`.\n"
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
            `MyDrive/wbsnet_kaggle_session2/paper_suite/kvasir/{A2,A5,A6,A7}/seed_3408/`

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
            "- Verify each `kvasir/<variant>/seed_3408/` folder contains\n"
            "  `checkpoints/best.pt` + `run_summary.json`.\n"
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
            `MyDrive/wbsnet_kaggle_session3/paper_suite/...`

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
            "- Verify all five run folders are populated on Drive.\n"
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
        "print(f'Drive root folder: MyDrive/{DRIVE_SESSION_NAME}/paper_suite/')\n"
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
