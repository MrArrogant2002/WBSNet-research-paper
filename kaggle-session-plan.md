# Kaggle Session Plan — WBSNet Paper Pipeline

> **Audience:** future Claude sessions (and the human researcher) working on the
> WBSNet IEEE journal paper. This file describes the active strategy for
> offloading work from Colab A100 onto Kaggle T4 free-tier sessions.
> **Status as of 2026-05-01.**

---

## 1. Why this exists

The Option-A paper run (12 configs × 3 seeds + ColonDB generalization +
aggregation) is estimated at **~645–890 A100 compute units** on Colab.
The user has **~581 compute units** available — short by ~60–310 units.

Each free Kaggle T4 session (12 h max wall-clock) absorbs roughly
**3–4 A100-equivalent hours** of work, i.e. **~40–55 A100 units saved per
session**. Three sessions therefore close the gap and leave headroom for
re-runs / smoke tests.

T4 vs A100 slowdown for this workload is empirically ~3–4× (256×256 inputs,
batch 8, ResNet-34 + wavelet ops).

---

## 2. Source of truth: Drive layout

```
MyDrive/
├── WBSNet_Dataset/                        # processed dataset (untouched)
│   ├── kvasir/{train,val,test}/{images,masks,boundaries}
│   ├── cvc_clinicdb/...
│   ├── cvc_colondb/...
│   └── isic2018/...
│
├── wbsnet_paper_runs/                     # ORIGINAL Kaggle artifacts (untouched)
│   └── paper_suite/
│       ├── kvasir/{A1..A7}/seed_3407/...
│       ├── cvc_clinicdb/{A1..A7}/seed_3407/...
│       └── isic2018/A1/seed_3407/...
│
├── wbsnet_kaggle_session1/                # NEW — Notebook 1 output
│   └── paper_suite/isic2018/A2/seed_3407/...
│
├── wbsnet_kaggle_session2/                # NEW — Notebook 2 output
│   └── paper_suite/kvasir/{A2,A5,A6,A7}/seed_3408/...
│
├── wbsnet_kaggle_session3/                # NEW — Notebook 3 output
│   └── paper_suite/
│       ├── kvasir/{A1,A3,A4}/seed_3408/...
│       └── cvc_clinicdb/{A1,A2}/seed_3408/...
│
└── WBSNet_outputs/                        # Colab A100 run output (still pending)
```

Each `paper_suite/<dataset>/<variant>/seed_<seed>/<run_name>/` folder mirrors
the layout produced by `train.py`:

```
checkpoints/best.pt              # ~99 MB
checkpoints/last.pt              # optional, only if resumed
metrics.csv
best_metrics.json
run_summary.json
resolved_config.json
evaluation/<dataset>_<split>.json
```

The Colab notebook imports every `wbsnet_kaggle_session*/` root via
`scripts/import_legacy_paper_runs.py`, which copies into `outputs/` and
verifies forward-pass compatibility.

---

## 3. Session-by-session plan

### Session 1 — `WBSNet_Kaggle_Session1.ipynb`

**Scope:** ISIC2018 A2 (full WBSNet) seed 3407.

| | |
|---|---|
| Configs | 1 |
| Train images | 2 594 |
| Per-epoch T4 | ~3–5 min |
| Total T4 wall | ~9–12 h |
| A100 saved | ~5–7 h ≈ 65–90 units |
| Drive output | `MyDrive/wbsnet_kaggle_session1/paper_suite/isic2018/A2/seed_3407/` |

**Why first.** Completes seed 3407 fully. After this, the seed-3407 column of
every paper table is "done"; the Colab run only needs to handle seeds 3408
and 3409.

**Risk:** if the session terminates before epoch 150, `best.pt` may still be
useful (it captures the best val Dice up to that point). The notebook
configures `train.save_every=10` so a partial run leaves a usable checkpoint.

### Session 2 — `WBSNet_Kaggle_Session2.ipynb`

**Scope:** Kvasir A2, A5, A6, A7 seed 3408 (the wavelet-relevant ablations).

| | |
|---|---|
| Configs | 4 |
| Train images | 880 each |
| Per-config T4 wall | ~2.5–3.5 h |
| Total T4 wall | ~10–14 h (TIGHT — see contingency) |
| A100 saved | ~6–8 h ≈ 78–104 units |
| Drive output | `MyDrive/wbsnet_kaggle_session2/paper_suite/kvasir/{A2,A5,A6,A7}/seed_3408/` |

**Why these four.** They directly exercise WBSNet components:
- **A2** = full WBSNet (headline ablation row)
- **A5** = WBSNet without boundary supervision
- **A6** = WBSNet without wavelet
- **A7** = WBSNet with db2 wavelet

Combined with the existing seed-3407 results, this gives **n=2** for the
significance tests on the most cited ablations.

**Contingency.** If the elapsed-time guard trips before A7 starts, A7 is
skipped and added to the Colab queue. The notebook prints a clear
`SKIPPED:` line for any unfinished config so future Claude can route them.

### Session 3 — `WBSNet_Kaggle_Session3.ipynb`

**Scope:** Kvasir A1, A3, A4 + ClinicDB A1, A2 seed 3408.

| | |
|---|---|
| Configs | 5 |
| Per-config T4 wall | Kvasir 2.5–3 h, ClinicDB 1.7–2.5 h |
| Total T4 wall | ~11–16 h (TIGHT) |
| A100 saved | ~6.5–9 h ≈ 85–117 units |
| Drive output | `MyDrive/wbsnet_kaggle_session3/paper_suite/...` |

**Why these five.** Closes out the Kvasir ablation table for seed 3408 (A1, A3, A4)
and gives both ClinicDB rows for seed 3408. Combined with sessions 1 + 2,
seed 3407 is 100% complete and seed 3408 is missing only ISIC2018 (A1, A2).

**Contingency.** Same as Session 2.

---

## 4. What's left for Colab A100 after all 3 sessions

Assuming all three Kaggle sessions complete cleanly:

| Bucket | Configs | A100 hours | Compute units |
|---|---|---|---|
| ISIC2018 A1, A2 seed 3408 | 2 | ~10–14 | ~130–182 |
| **Optional** seed 3409 (full 11 configs) | 11 | ~22–31 | ~290–400 |
| ColonDB generalization (inference) | n/a | <1 | <13 |
| **Total if seed 3409 included** | | ~33–46 | **~420–580** |
| **Total if seed 3409 dropped (n=2 paper)** | | ~11–15 | **~143–195** |

Recommendation: **run seed 3408 ISIC2018 on Colab, defer seed 3409**.
Submit with n=2; if reviewers push back, run seed 3409 in a follow-up
Kaggle pass and update the tables.

---

## 5. One-time prerequisites

### 5.1 Google Drive OAuth token (Kaggle Secrets)

Each Kaggle notebook authenticates to Drive using **OAuth refresh-token
credentials** stored as Kaggle Secrets. Setup steps:

1. Open <https://console.cloud.google.com/>, create or pick a project.
2. Enable **Google Drive API**.
3. **APIs & Services → Credentials → Create Credentials → OAuth client ID**.
   Application type: **Desktop app**. Download the JSON.
4. On a local machine (Python ≥ 3.10):
   ```bash
   pip install google-auth-oauthlib google-api-python-client
   python -c "
   from google_auth_oauthlib.flow import InstalledAppFlow
   flow = InstalledAppFlow.from_client_secrets_file(
       'credentials.json', ['https://www.googleapis.com/auth/drive']
   )
   creds = flow.run_local_server(port=0)
   print('CLIENT_ID:', creds.client_id)
   print('CLIENT_SECRET:', creds.client_secret)
   print('REFRESH_TOKEN:', creds.refresh_token)
   "
   ```
5. In the Kaggle notebook UI: **Add-ons → Secrets**, add three secrets:
   - `GDRIVE_CLIENT_ID`
   - `GDRIVE_CLIENT_SECRET`
   - `GDRIVE_REFRESH_TOKEN`

The same three secrets work for all three notebooks; do this once.

### 5.2 Processed dataset on Kaggle

Two options. Pick whichever fits the workflow:

- **(Preferred) Kaggle Dataset.** Upload the processed `WBSNet_Dataset/`
  folder as a private Kaggle Dataset (e.g. slug `wbsnet-processed`). Attach
  it to each notebook. Reads from `/kaggle/input/wbsnet-processed/...`
  (local SSD, ~10× faster than Drive on Kaggle).
- **(Fallback) Drive mount.** Slower but does not require a Kaggle Dataset
  upload. Each notebook contains the fallback code commented out.

The notebooks default to the Kaggle Dataset path; switch the dataset cell
if using the Drive fallback.

### 5.3 Repo

The notebooks clone <https://github.com/MrArrogant2002/WBSNet-research-paper>.
Make sure the latest stability fixes (commit `cd5d3f1` or newer) are on
`main` — the Kaggle runs depend on `train.nonfinite_grad_action=skip` and
`requirements-colab.txt`.

---

## 6. Operating procedure (for the human)

1. **Before each session:** confirm Kaggle T4 quota in
   <https://www.kaggle.com/settings/account>. Free tier resets weekly.
2. **Open the matching `WBSNet_Kaggle_Session{N}.ipynb` on Kaggle.**
3. **Runtime → Settings → Accelerator → GPU T4 ×1.** Internet ON.
4. **Attach** the `wbsnet-processed` Kaggle Dataset (or set the Drive path).
5. **Run all cells.** The notebook will:
   - verify T4
   - clone the repo
   - install deps
   - authenticate Drive via secrets
   - run a 1-epoch smoke test
   - launch the scope loop (skips a config if elapsed time would exceed 11.5 h)
   - upload outputs to Drive every 10 min
   - flush a final time before exit
6. **After session ends:** open
   `https://drive.google.com/drive/folders/...wbsnet_kaggle_session{N}` and
   verify each variant has `checkpoints/best.pt` + `run_summary.json`.

---

## 7. After all 3 sessions: Colab handoff

Run `WBSNet_Colab.ipynb` with:

```
--seeds 3407 3408
```

(omit 3409 unless you have spare units). Section 6 of that notebook now
imports from **all four** Drive roots (`wbsnet_paper_runs`,
`wbsnet_kaggle_session1`, `wbsnet_kaggle_session2`, `wbsnet_kaggle_session3`),
so every completed run is skipped automatically.

---

## 8. Future work / open questions

- **Seed 3409.** If reviewers ask for n=3, run seed 3409 either on a fourth
  Kaggle pass (split across 2–3 sessions) or on a fresh Colab budget.
- **Lambda sweep.** The boundary-loss sensitivity figure
  (`paper/figures/lambda_sensitivity.png`) is not yet computed. Plan: one
  more Kaggle session running `scripts/plot_lambda_sweep.py` after a small
  λ ∈ {0.0, 0.25, 0.5, 0.75, 1.0} sweep on Kvasir A2 seed 3407 (5 short
  runs × 50 epochs ≈ 5 h T4).
- **Significance tests.** Once seed 3408 is complete on every config,
  `scripts/significance_tests.py` produces the paired-test table for the
  paper. This step runs on CPU; no GPU session needed.
- **Qualitative figure.** `scripts/make_paper_figures.py` is run from Colab
  Section 10; uses Kvasir A2's `predictions/` directory.

---

## 9. Quick reference — variant → config map

| Variant | Description | Config file |
|---|---|---|
| A1 | Identity U-Net (no wavelet, no LFSA, no HFBA) | `configs/ablation_identity_unet.yaml` |
| A2 | Full WBSNet (haar) | `configs/kvasir_wbsnet.yaml` (or `clinicdb_wbsnet.yaml`, `isic2018_wbsnet.yaml`) |
| A3 | Wavelet + LFSA only | `configs/ablation_lfsa_only.yaml` |
| A4 | Wavelet + HFBA only | `configs/ablation_hfba_only.yaml` |
| A5 | Full WBSNet, boundary loss disabled | `configs/ablation_no_boundary_supervision.yaml` |
| A6 | LFSA + HFBA, no wavelet | `configs/ablation_no_wavelet.yaml` |
| A7 | Full WBSNet, db2 wavelet | `configs/ablation_db2_wavelet.yaml` |

ClinicDB / ISIC2018 baselines (A1) use
`configs/clinicdb_unet_baseline.yaml` / `configs/isic2018_unet_baseline.yaml`.

---

## 10. Contact / handoff

Owner: Rapolu Eswara Balu (`balu123456sbb@gmail.com`).
Last updated: 2026-05-01 (Claude Opus 4.7 session).
