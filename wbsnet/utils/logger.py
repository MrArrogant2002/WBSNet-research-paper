from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .io import ensure_dir, flatten_dict


class ExperimentLogger:
    def __init__(
        self,
        output_dir: str | Path,
        config: dict[str, Any],
        enabled: bool,
        rank: int,
        open_csv: bool = True,
        append_csv: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.rank = rank
        self.csv_path = self.output_dir / "metrics.csv"
        self.csv_file = None
        self.writer = None
        self.wandb_run = None
        self.wandb_module = None
        if rank == 0 and (open_csv or enabled):
            ensure_dir(self.output_dir)
        if rank == 0 and open_csv:
            file_exists = append_csv and self.csv_path.exists() and self.csv_path.stat().st_size > 0
            mode = "a" if append_csv else "w"
            self.csv_file = self.csv_path.open(mode, encoding="utf-8", newline="")
            self._csv_has_header = file_exists
        else:
            self._csv_has_header = False
        if enabled and rank == 0:
            self._init_wandb(config)

    def _init_wandb(self, config: dict[str, Any]) -> None:
        runtime = config.get("runtime", {}).get("wandb", {})
        if not runtime.get("enabled", False):
            return
        try:
            import os
            import wandb

            api_key = os.environ.get("WANDB_API_KEY")
            if api_key:
                wandb.login(key=api_key, relogin=False)
            self.wandb_module = wandb
            self.wandb_run = wandb.init(
                project=runtime.get("project", "WBSNet"),
                entity=runtime.get("entity"),
                name=config.get("experiment", {}).get("run_name"),
                config=config,
                dir=str(self.output_dir),
                mode=runtime.get("mode", "online"),
                tags=config.get("experiment", {}).get("tags", []),
            )
        except Exception as exc:
            print(f"[W&B] disabled: {exc}")
            self.wandb_run = None

    def log_metrics(self, step: int, metrics: dict[str, Any]) -> None:
        if self.rank != 0:
            return

        row = {"step": step}
        row.update(flatten_dict(metrics))
        if self.writer is None and self.csv_file is not None:
            self.writer = csv.DictWriter(self.csv_file, fieldnames=list(row.keys()))
            if not self._csv_has_header:
                self.writer.writeheader()

        if self.writer is not None:
            self.writer.writerow(row)
            self.csv_file.flush()

        if self.wandb_run is not None:
            self.wandb_run.log(row, step=step)

    def log_images(self, payload: dict[str, Any], step: int) -> None:
        if self.rank != 0 or self.wandb_run is None:
            return
        try:
            self.wandb_run.log(payload, step=step)
        except Exception as exc:
            print(f"[W&B] image logging skipped: {exc}")

    def log_panel_images(self, key: str, images: list[dict[str, Any]], step: int) -> None:
        if self.rank != 0 or self.wandb_run is None or self.wandb_module is None or not images:
            return
        try:
            payload = {
                key: [
                    self.wandb_module.Image(item["image"], caption=item.get("caption"))
                    for item in images
                ]
            }
            self.wandb_run.log(payload, step=step)
        except Exception as exc:
            print(f"[W&B] panel logging skipped: {exc}")

    def finish(self) -> None:
        if self.rank == 0 and self.csv_file is not None:
            self.csv_file.close()
        if self.wandb_run is not None:
            self.wandb_run.finish()
