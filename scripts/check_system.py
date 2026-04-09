from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _run_command(command: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except FileNotFoundError:
        return {"command": command, "returncode": 127, "stdout": "", "stderr": "command not found"}


def _disk_usage(path: str) -> dict[str, float]:
    total, used, free = shutil.disk_usage(path)
    gib = 1024**3
    return {
      "total_gb": round(total / gib, 2),
      "used_gb": round(used / gib, 2),
      "free_gb": round(free / gib, 2),
    }


def _python_packages() -> dict[str, str]:
    packages: dict[str, str] = {}
    for module_name in [
        "torch",
        "torchvision",
        "numpy",
        "pandas",
        "PIL",
        "yaml",
        "wandb",
        "pywt",
        "pytorch_wavelets",
        "albumentations",
        "cv2",
        "scipy",
        "segmentation_models_pytorch",
        "matplotlib",
    ]:
        try:
            module = __import__(module_name)
            packages[module_name] = getattr(module, "__version__", "installed")
        except Exception as exc:
            packages[module_name] = f"missing: {type(exc).__name__}"
    return packages


def _torch_report() -> dict[str, Any]:
    try:
        import torch

        report: dict[str, Any] = {
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "device_count": int(torch.cuda.device_count()),
            "nccl_available": bool(torch.distributed.is_nccl_available()),
            "cudnn_enabled": bool(torch.backends.cudnn.enabled),
        }
        devices = []
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_gb": round(properties.total_memory / (1024**3), 2),
                    "multi_processor_count": properties.multi_processor_count,
                }
            )
        report["devices"] = devices
        return report
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    output_dir = Path("artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "hostname": platform.node(),
        },
        "environment": {
            "cwd": os.getcwd(),
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "disk": {
            "workspace": _disk_usage("."),
            "tmp": _disk_usage("/tmp"),
        },
        "packages": _python_packages(),
        "torch": _torch_report(),
        "commands": {
            "nvidia_smi": _run_command(["nvidia-smi"]),
            "nvidia_smi_query": _run_command(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,driver_version,temperature.gpu,utilization.gpu",
                    "--format=csv,noheader",
                ]
            ),
            "nvcc_version": _run_command(["nvcc", "--version"]),
        },
    }

    output_path = output_dir / "system_report.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved system report to {output_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
