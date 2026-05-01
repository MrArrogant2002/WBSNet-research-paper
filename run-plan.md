# WBSNet Run Plan — Operational Schedule

> **Audience:** the human researcher (and any Claude session resuming the
> work). This file is the single-page operational schedule for
> all training, evaluation, and aggregation needed to ship the IEEE
> journal submission. **Status: 2026-05-01.**
>
> Read together with `kaggle-session-plan.md` (the *why* behind the Kaggle
> split) and `paper/paper-outline.md` (the *what* the numbers feed into).

---

## 1. TL;DR

| Phase | Where | Wall time | Cost | Output unlocked |
|---|---|---|---|---|
| **K1.** Complete seed 3407 | Kaggle T4 | ~9–12 h | free (T4 quota) | seed 3407 100 % done |
| **K2.** Kvasir seed 3408 (wavelet ablations) | Kaggle T4 | ~10–14 h | free | A2/A5/A6/A7 row of seed 3408 |
| **K3.** Kvasir + ClinicDB seed 3408 (rest) | Kaggle T4 | ~11–16 h | free | seed 3408 minus ISIC2018 |
| **C1.** ISIC2018 + aggregation + figures | Colab A100 | ~13–18 h | **~165–235 units** | full Option-A tables |
| **Buffer / follow-up** | Colab | optional | 0–400 units | seed 3409 if reviewers ask |

**Compute units consumed in the headline plan: ~165–235 of 581 available** — leaves ~350–415 unit margin for re-runs, smoke tests, or seed 3409.

---

## 2. Compute budget

| Resource | Available | Allocated to plan | Reserve |
|---|---|---|---|
| Colab Pro+ A100 compute units | 581 | ~165–235 (28–40 %) | ~350–415 |
| Kaggle free T4 hours / week | ~30 | 30–42 (over 1–2 weeks) | weekly auto-reset |
| Local researcher time | n/a | ~6–8 h supervised + 50–60 h background | n/a |

Slowdown ratio: T4 vs A100 ≈ 3–4 × for this workload. 1 hour A100 ≈ 13 compute units.

---

## 3. Master timeline

The four sessions are **strictly sequential** because each one's output is consumed by the next. Plan ~5–8 calendar days end-to-end:

```
Day 0 (prep)  ─┐
                ├─ One-time: Drive OAuth, Kaggle Dataset upload
Day 1         ─┤
                ├─ K1   Kaggle Session 1                          (12 h)
                │
Day 2 (rest)  ─┤  Kaggle quota recharge, verify K1 output on Drive
Day 3         ─┤
                ├─ K2   Kaggle Session 2                          (12 h)
                │
Day 4 (rest)  ─┤  verify K2
Day 5         ─┤
                ├─ K3   Kaggle Session 3                          (12 h)
                │
Day 6 (rest)  ─┤  verify K3
Day 7         ─┤
                └─ C1   Colab paper run + aggregation + figures    (~13–18 h)
```

If your weekly T4 quota is fresh, K1+K2 can run on consecutive days (skipping the rest day) without stalling.

---

## 4. Phase-by-phase cost detail

### Phase K1 — Kaggle Session 1

**Notebook:** [WBSNet_Kaggle_Session1.ipynb](WBSNet_Kaggle_Session1.ipynb)
**Scope:** ISIC2018 A2 seed 3407 (1 config)

| Field | Value |
|---|---|
| Configs trained | 1 |
| Epochs | 150 |
| Batch / accum | 8 / 2 (eff. 16) |
| T4 wall time (est.) | 9–12 h |
| T4 GPU-hours used | 9–12 of weekly 30 |
| Colab units saved | ~65–90 |
| Drive output | `MyDrive/wbsnet_kaggle_session1/paper_suite/isic2018/A2/seed_3407/` |
| Closes the gap on | Seed 3407 ISIC2018 A2 |

### Phase K2 — Kaggle Session 2

**Notebook:** [WBSNet_Kaggle_Session2.ipynb](WBSNet_Kaggle_Session2.ipynb)
**Scope:** Kvasir A2, A5, A6, A7 seed 3408 (4 configs)

| Field | Value |
|---|---|
| Configs trained | 4 |
| Epochs each | 150 |
| Per-config T4 wall | 2.5–3.5 h |
| Total T4 wall | 10–14 h (TIGHT — time guard may defer A7) |
| Colab units saved | ~78–104 |
| Drive output | `MyDrive/wbsnet_kaggle_session2/paper_suite/kvasir/{A2,A5,A6,A7}/seed_3408/` |
| Closes the gap on | Wavelet-relevant ablation rows of seed 3408 |

### Phase K3 — Kaggle Session 3

**Notebook:** [WBSNet_Kaggle_Session3.ipynb](WBSNet_Kaggle_Session3.ipynb)
**Scope:** Kvasir A1/A3/A4 + ClinicDB A1/A2 seed 3408 (5 configs)

| Field | Value |
|---|---|
| Configs trained | 5 |
| Epochs each | 150 |
| Kvasir per-config T4 | 2.5–3.5 h × 3 |
| ClinicDB per-config T4 | 1.7–2.5 h × 2 |
| Total T4 wall | 11–16 h (TIGHT — last 1–2 may defer to Colab) |
| Colab units saved | ~85–117 |
| Drive output | `MyDrive/wbsnet_kaggle_session3/paper_suite/...` |
| Closes the gap on | Remaining seed 3408 except ISIC2018 |

### Phase C1 — Colab paper run

**Notebook:** [WBSNet_Colab.ipynb](WBSNet_Colab.ipynb)
**Scope:** ISIC2018 A1+A2 seed 3408, ColonDB generalization, aggregation, significance, complexity, qualitative figures.

| Step | Section | A100 hours | Units (~13/hr) |
|---|---|---|---|
| Drive mount + dependency install + dataset link | 5 | 0.05–0.1 | 1 |
| Restore + 4-root legacy import | 6 | 0.1 | 1–2 |
| 1-epoch smoke test | 8 | 0.05 | <1 |
| ISIC2018 A1 seed 3408 (150 epochs) | 9 | 5–7 | **65–91** |
| ISIC2018 A2 seed 3408 (150 epochs) | 9 | 5–7 | **65–91** |
| ColonDB generalization eval (inference only, both seeds) | 9 | 0.2–0.4 | 3–6 |
| Aggregation + significance tests + complexity | 9 (post-train) | <0.2 | <3 |
| Qualitative figures (predict + grid) | 10 | 0.3–0.6 | 4–8 |
| **Phase total** | | **~10.9–15.4 h** | **~140–203** |

Add ~10–15 % overhead for Colab disconnects / warm-ups / re-runs:

| Phase C1 with overhead | **~13–18 h A100** | **~165–235 units** |
|---|---|---|

Drops well inside the 581-unit budget with ~350–415 units of margin.

---

## 5. Total cost rollup

| | Wall hours | A100-equivalent | Compute units |
|---|---|---|---|
| K1 | 9–12 (T4) | 3–4 | 0 (free Kaggle) |
| K2 | 10–14 (T4) | 3–4 | 0 |
| K3 | 11–16 (T4) | 3–4 | 0 |
| C1 | 13–18 (A100) | 13–18 | 165–235 |
| **Total — headline plan** | **43–60** | **22–30** | **165–235 / 581** |
| Reserve (re-runs, smoke, seed 3409) | – | – | 350–415 |

If seed 3409 is later required (n=3 paper):

| Add-on | Wall hours | Units |
|---|---|---|
| Seed 3409, full 11 configs on Colab | 22–31 (A100) | 290–400 |
| Seed 3409, partial (Kvasir only) | 10–14 (A100) | 130–180 |
| Seed 3409 distributed: 7 Kvasir on Kaggle + ISIC + ClinicDB on Colab | 14–20 (T4) + ~12 (A100) | ~155–200 |

---

## 6. Pre-flight checklist (do once before K1)

- [ ] Google Cloud project + Drive API enabled
- [ ] OAuth client (Desktop app) created; refresh token generated locally
- [ ] Kaggle Secrets set: `GDRIVE_CLIENT_ID`, `GDRIVE_CLIENT_SECRET`, `GDRIVE_REFRESH_TOKEN`
- [ ] Processed `WBSNet_Dataset/` uploaded as Kaggle Dataset (suggested slug `wbsnet-processed`)
- [ ] Repo `main` is at commit ≥ `cd5d3f1` (stability fixes present)
- [ ] Confirmed weekly Kaggle T4 quota ≥ 12 h before starting K1
- [ ] (Colab) `WANDB_API_KEY` set in Colab Secrets if online W&B is desired
- [ ] (Colab) Pro+ subscription active and budget verified at 581 units

## 7. Per-session pre-flight (do before each Kaggle run)

- [ ] Open the right notebook (K1 / K2 / K3)
- [ ] Settings → Accelerator → **GPU T4 ×1**
- [ ] Settings → Internet **ON**
- [ ] Add data → attach `wbsnet-processed`
- [ ] Run all cells; confirm smoke test passes before walking away
- [ ] Open `https://drive.google.com/drive/folders/<session-folder>` in another tab to watch uploads land

## 8. Mid-flight monitoring

- **Kaggle:** the notebook prints `[DriveSyncer] uploaded N file(s)` every 10 minutes. If N is consistently 0 after epoch 10, something is wrong with the uploader.
- **Colab Section 9:** tail `/content/rsync.log`. If it stops growing, the periodic syncer crashed; restart it.
- **GPU utilization:** both T4 and A100 should sit at ≥ 85 % during convolutions. If you see ≤ 50 %, the dataloader is the bottleneck — bump `num_workers` to 4.

## 9. Post-training aggregation (Colab Section 9 tail + manual)

After C1's training portion finishes, the runner emits:

```
outputs/aggregated/aggregated_results.csv
outputs/aggregated/aggregated_results.json
outputs/significance/<...>.json
outputs/model_complexity/<...>.json
```

Manual follow-ups:

- [ ] `python aggregate_results.py --root outputs --output outputs/aggregated`
- [ ] `python scripts/significance_tests.py --root outputs --output outputs/significance --record-type evaluation --reference A1_identity_unet`
- [ ] `python scripts/model_complexity.py --output outputs/model_complexity`
- [ ] Generate qualitative panels (Section 10 of Colab notebook)
- [ ] Lambda sweep (separate Kaggle session, see Section 12 below)
- [ ] Update `paper/paper.tex` Result Tables 1–3 and Figures 3–4
- [ ] Run `/paper-polish` once all numbers are in
- [ ] Run `/plagiarism-check` before submission

## 10. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Kaggle session disconnects mid-run | M | L | DriveSyncer uploads every 10 min; partial best.pt usable; restart picks up via `best_pt_exists` skip |
| Weekly Kaggle quota exhausted | M | M | Spread sessions across 2 weeks if needed; quota resets Mondays UTC |
| Colab disconnects during ISIC2018 | M | M | Colab Section 9 background rsync to Drive; resume from `last.pt` is supported via runner |
| OOM on T4 (16 GB) at batch 8 | L | M | Drop to batch 4, raise grad_accum_steps to 4 (eff. 16 unchanged) |
| OAuth refresh token expires (rare; > 6 mo unused) | L | L | Re-run local OAuth script, update Kaggle Secrets |
| Mixed-hardware std inflation on ISIC2018 row | H (by design) | L | Disclosed in paper Experimental Setup; balanced design (1 T4 + 1 A100 on both A1 and A2) keeps comparisons unbiased |
| Reviewer demands n=3 | M | M | Run seed 3409 in follow-up (cost row in §5) |

## 11. Quick reference — what's where on Drive

```
MyDrive/
├── WBSNet_Dataset/               processed dataset (read-only)
├── wbsnet_paper_runs/            legacy seed-3407 artifacts (read-only)
├── wbsnet_kaggle_session1/       K1 output → ISIC2018 A2 seed 3407
├── wbsnet_kaggle_session2/       K2 output → Kvasir A2/A5/A6/A7 seed 3408
├── wbsnet_kaggle_session3/       K3 output → Kvasir A1/A3/A4 + ClinicDB seed 3408
└── WBSNet_outputs/               C1 output + final aggregations
```

## 12. Open follow-ups (after the headline plan)

| Task | Where | Estimated cost |
|---|---|---|
| Lambda sensitivity sweep (5 λ × 50 epochs on Kvasir A2 seed 3407) | Kaggle T4, 1 session | ~5–6 h T4 (free) |
| Plot lambda sensitivity → `paper/figures/lambda_sensitivity.png` | local CPU | < 1 min |
| WBS module TikZ block diagram (Fig. 2) | manual / TikZ | ~30 min researcher time |
| Seed 3409 run (if reviewers request n=3) | Kaggle + Colab mix | ~155–200 units / ~14–20 h T4 |
| Paper polish + plagiarism check | local | ~30 min |
| Submission packaging (Overleaf zip + cover letter) | local | ~30 min |

## 13. Done definition

Submission-ready when:

- [ ] All entries in `aggregated_results.csv` for seeds 3407+3408 across every (dataset, variant) cell
- [ ] All paired-test p-values for ablation comparisons computed
- [ ] Tables 1–3 in `paper/paper.tex` filled with mean ± std and significance markers
- [ ] Fig. 3 (qualitative grid) embedded
- [ ] Fig. 4 (lambda sensitivity) embedded
- [ ] Fig. 2 (WBS module TikZ) embedded
- [ ] Hardware-mix disclosure sentence added to Experimental Setup
- [ ] paper-polish + plagiarism-check passes
- [ ] Overleaf compiles cleanly with `IEEEtran`

When that checklist is green, hand off to the venue submission flow.

---

Last updated: 2026-05-01 (Claude Opus 4.7 session).
