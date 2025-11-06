"""
UrbanSim WM - FastAPI Backend
Main application entry point with CORS and routing setup
"""

import logging
import os
import subprocess
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from app.api import routes
from app.core.config import settings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("UrbanSim WM API starting up...")
    logger.info(f"Environment: {settings.ENVIRONMENT}")

    # Optional ETL bootstrap (non-blocking, with log streaming)
    try:
        flag = str(getattr(settings, "RUN_ETL_ON_STARTUP", "true")).lower()
        if flag in (
            "1",
            "true",
            "yes",
        ):

            def _run_bootstrap():
                try:
                    cities = [
                        c.strip() for c in settings.ETL_CITIES.split(",") if c.strip()
                    ]
                    hours = int(getattr(settings, "ETL_OPENAQ_HOURS", 24))
                    logger.info(
                        f"ETL bootstrap enabled. Cities={cities}, hours={hours}"
                    )

                    logs_dir = Path(settings.LOGS_DIR)
                    logs_dir.mkdir(parents=True, exist_ok=True)
                    job_id = str(uuid.uuid4())[:8]
                    log_file = logs_dir / f"bootstrap_{job_id}.log"
                    logger.info(
                        f"ETL bootstrap job_id={job_id} - streaming logs to {log_file}"
                    )
                    # Try WAQI first (more reliable), fallback to OpenAQ
                    try:
                        from app.etl.waqi import write_recent_pm25_mean

                        etl_source = "WAQI"
                    except ImportError:
                        from app.etl.openaq import write_recent_pm25_mean

                        etl_source = "OpenAQ"

                    for city in cities:
                        with open(log_file, "a", encoding="utf-8", buffering=1) as f:
                            line = f"\n=== [{datetime.utcnow().isoformat()}Z] Air Quality ETL start city={city} hours={hours} (source={etl_source}) ===\n"
                            f.write(line)
                            f.flush()
                            try:
                                out_path = write_recent_pm25_mean(
                                    city=city,
                                    hours=hours,
                                    out_dir=settings.PROCESSED_DATA_DIR,
                                    log_path=str(log_file),
                                )
                                msg = f"Air Quality ETL completed city={city} -> {out_path} (source={etl_source})\n"
                                f.write(msg)
                                f.flush()
                                logger.info(msg.strip())
                            except Exception as ex2:
                                err = f"Air Quality ETL failed city={city}: {ex2}\n"
                                f.write(err)
                                f.flush()
                                logger.warning(err.strip())
                        logger.info(
                            f"ETL bootstrap job_id={job_id} completed. See logs: {log_file}"
                        )
                except Exception as ex:
                    logger.warning(f"ETL bootstrap failed: {ex}")

            # Run ETL with a startup timeout: stream logs, but don't block indefinitely
            t = threading.Thread(target=_run_bootstrap, daemon=True)
            t.start()
            timeout = int(getattr(settings, "ETL_STARTUP_TIMEOUT_SECS", 10))
            t.join(timeout=timeout)
            if t.is_alive():
                logger.info(
                    f"ETL bootstrap still running after {timeout}s; continuing startup. Logs at /logs/."
                )
        else:
            logger.info(
                "ETL bootstrap disabled (set RUN_ETL_ON_STARTUP=true to enable)."
            )
    except Exception as e:
        logger.warning(f"ETL bootstrap skipped due to error: {e}")

    try:
        from app.models.model_wrapper import get_model

        if str(getattr(settings, "WARM_MODEL_ON_STARTUP", "true")).lower() in (
            "1",
            "true",
            "yes",
        ):
            logger.info("Warming up model on startup...")
            get_model(settings.MODEL_CHECKPOINT_PATH)
            logger.info("Model warmed up successfully")
    except Exception as e:
        logger.warning(f"Model warm-up skipped due to error: {e}")

    yield
    logger.info("UrbanSim WM API shutting down...")


# Initialize FastAPI application
app = FastAPI(
    title="UrbanSim WM API",
    version="0.1.0",
    description="Smart City World Model - API for simulating urban dynamics",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(routes.router, prefix="/api")


@app.get("/")
def root():
    """Root endpoint - health check"""
    return {
        "status": "ok",
        "service": "UrbanSim WM API",
        "version": "0.1.0",
        "docs": "/docs",
    }


## Deprecated on_event handlers removed in favor of lifespan above
