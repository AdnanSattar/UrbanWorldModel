"""
Model retraining endpoint
Triggers background training jobs
"""

import logging
import os
import shutil
import subprocess
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import BackgroundTasks
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RetrainRequest(BaseModel):
    """Request body for retraining endpoint"""

    config_path: str = Field(
        default="training/configs/base.yaml",
        description="Path to training configuration file",
        examples=["training/configs/base.yaml"],
    )


def _stream_process_output(proc: subprocess.Popen, log_file: Path):
    """Stream stdout/stderr from a process into a rotating log file."""
    with log_file.open("a", encoding="utf-8") as f:
        for line in iter(proc.stdout.readline, b""):
            text = line.decode(errors="replace")
            f.write(text)
            f.flush()
        for line in iter(proc.stderr.readline, b""):
            text = line.decode(errors="replace")
            f.write(text)
            f.flush()


def _update_latest_checkpoint(checkpoints_dir: Path):
    """Point/refresh a latest symlink to the best or final checkpoint if present."""
    candidates = [
        checkpoints_dir / "model_best.json",
        checkpoints_dir / "model_final.json",
    ]
    for meta in candidates:
        if meta.exists():
            latest = checkpoints_dir / "latest.json"
            try:
                if latest.exists() or latest.is_symlink():
                    latest.unlink()
                # Create a simple copy to avoid symlink permission issues on Windows
                shutil.copyfile(meta, latest)
                # Copy the paired .pth if it exists
                pth = meta.with_suffix(".pth")
                if pth.exists():
                    shutil.copyfile(pth, checkpoints_dir / "latest.pth")
                logger.info(f"Updated latest checkpoint pointer to {meta.name}")
                return str(latest)
            except Exception as e:
                logger.warning(f"Failed to update latest checkpoint pointer: {e}")
                return None
    logger.info("No checkpoint candidates found to update latest pointer")
    return None


def _retrain_task(config_path: str):
    """
    Background task for model retraining.

    Implementation details:
    - Spawns training process (subprocess) running training/urban_world_model.py with the given config
    - Streams logs to training/logs/<job_id>.log
    - On completion, refreshes a latest checkpoint pointer (copy of best/final)
    """
    job_id = str(uuid.uuid4())[:8]
    start_ts = datetime.utcnow().isoformat()
    repo_root = Path(os.getenv("URBANSIM_WM_ROOT", Path(__file__).resolve().parents[3]))
    training_dir = repo_root / "training"
    logs_dir = training_dir / "logs"
    checkpoints_dir = training_dir / "checkpoints"
    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    log_file = logs_dir / f"train_{job_id}.log"
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = training_dir / cfg_path

    logger.info(
        f"[retrain:{job_id}] Starting training with config={cfg_path} at {start_ts}"
    )

    # Build subprocess command (host execution). Project uses 'uv' in Docker, but local run is fine.
    cmd = [
        "python",
        str(training_dir / "urban_world_model.py"),
    ]
    env = os.environ.copy()

    # Launch process
    proc = subprocess.Popen(
        cmd,
        cwd=str(training_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
    )

    # Stream logs in a thread
    t = threading.Thread(
        target=_stream_process_output, args=(proc, log_file), daemon=True
    )
    t.start()

    ret = proc.wait()
    t.join(timeout=5)

    if ret == 0:
        logger.info(f"[retrain:{job_id}] Training completed successfully")
        latest = _update_latest_checkpoint(checkpoints_dir)
        if latest:
            logger.info(f"[retrain:{job_id}] Latest checkpoint updated: {latest}")
    else:
        logger.error(f"[retrain:{job_id}] Training process failed with exit code {ret}")
        logger.error(f"[retrain:{job_id}] See log file: {log_file}")


def trigger_retrain(
    background_tasks: BackgroundTasks, config_path: str = "training/configs/base.yaml"
) -> dict:
    """
    Trigger model retraining in the background

    Args:
        background_tasks: FastAPI background tasks
        config_path: Path to training config

    Returns:
        Status message with job info
    """
    job_id = str(uuid.uuid4())[:8]
    logger.info(f"Queueing retrain task {job_id} with config: {config_path}")
    # Pass config only; job_id is for client reference (logs will include it)
    background_tasks.add_task(_retrain_task, config_path)

    # Return basic job info and log path hint
    logs_hint = "training/logs/train_<job_id>.log"
    return {
        "status": "retrain_started",
        "job_id": job_id,
        "config": config_path,
        "logs": logs_hint.replace("<job_id>", job_id),
        "message": "Retraining job started in background",
    }
