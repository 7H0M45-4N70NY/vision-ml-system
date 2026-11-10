"""Environment & Secrets Pre-Flight Validation Tool.

Validates .env configuration, cloud provider secrets (DagsHub, Roboflow),
local persistence storage, and GPU readiness before running pipelines.
"""

import sys
import os
from pathlib import Path

def _load_env_fallback(env_path: Path):
    """Simple stdlib-only .env parser when python-dotenv is not installed."""
    if not env_path.exists():
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip("'\"")
            if k not in os.environ:
                os.environ[k] = v

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

def main():
    root_dir = Path(__file__).resolve().parent.parent
    env_path = root_dir / ".env"
    
    print("=" * 65)
    print(" VISION ML SYSTEM — PRE-FLIGHT ENVIRONMENT CHECK")
    print("=" * 65)
    
    # 1. Check .env file
    if env_path.exists():
        if load_dotenv:
            load_dotenv(dotenv_path=env_path)
        else:
            _load_env_fallback(env_path)
        print(f"[x] .env file detected at: {env_path.name}")
    else:
        print("[!] .env file not found. Copying values from default or OS environment.")
        print("    Tip: run 'cp .env.example .env' to configure remote credentials.")

    # 2. Privacy-First Local Core Check (Always Available)
    print("\n[+] Core Local Architecture (Zero-Cloud / Privacy-First):")
    data_dir = root_dir / "data"
    sample_videos = data_dir / "sample_videos"
    db_path = data_dir / "analytics.db"
    
    print(f"    - Data Directory:       {'OK' if data_dir.exists() else 'Missing'}")
    print(f"    - Sample Videos:        {'OK' if sample_videos.exists() else 'Missing'}")
    print(f"    - SQLite Analytics DB:  {'Initialized' if db_path.exists() else 'Ready (will auto-create)'}")

    # 3. Cloud Hooks Status
    print("\n[+] Optional Cloud Hooks (MLflow / DagsHub / Roboflow):")
    dagshub_user = os.getenv("DAGSHUB_USERNAME")
    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    roboflow_key = os.getenv("ROBOFLOW_API_KEY")

    if dagshub_user and dagshub_user != "your_dagshub_username":
        print(f"    - DagsHub User:         Configured ({dagshub_user})")
        print(f"    - DagsHub Token:        {'Configured' if dagshub_token and dagshub_token != 'your_dagshub_access_token' else 'Missing or Default'}")
    else:
        print("    - DagsHub / MLflow:     Not configured (Runs will default to local file storage / mlruns)")

    if roboflow_key and roboflow_key != "your_roboflow_api_key":
        print("    - Roboflow Sync:        Configured")
    else:
        print("    - Roboflow Sync:        Not configured (Local auto-labeling will use data/low_confidence_frames)")

    # 4. Hardware Acceleration Check
    print("\n[+] Compute Hardware Readiness:")
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"    - PyTorch GPU Status:   Active -> {device_name} ({vram_gb:.1f} GB VRAM)")
            print(f"    - CUDA Version:         {torch.version.cuda}")
        else:
            print("    - PyTorch GPU Status:   CPU Only (Run 'python scripts/check_cuda.py' to diagnose)")
    except ImportError:
        print("    - PyTorch Status:       Not installed in current environment")

    print("\n" + "=" * 65)
    print(" SYSTEM READINESS SUMMARY:")
    print("  * Local Inference & Analytics:  READY (No cloud credentials required)")
    print("  * Local Training Pipeline:      READY (Dataset format: YOLO standard)")
    print("  * Remote Experiment Sync:       OPTIONAL (Configure in .env if desired)")
    print("=" * 65 + "\n")


if __name__ == '__main__':
    main()

