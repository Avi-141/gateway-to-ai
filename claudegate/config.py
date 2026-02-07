"""Configuration constants and logging setup."""

import logging.config
import os

# Server defaults
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8080
DEFAULT_LOG_LEVEL = "INFO"

# AWS defaults
DEFAULT_AWS_REGION = "us-west-2"
DEFAULT_BEDROCK_REGION_PREFIX = "us"
DEFAULT_READ_TIMEOUT = 300  # 5 minutes for slow models like Opus

# Logging
LOG_LEVEL = os.environ.get("LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()

# Unified logging configuration using uvicorn's ColourizedFormatter
# This ensures both uvicorn and app logs have the same rich format with colors
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "()": "uvicorn.logging.ColourizedFormatter",
            "fmt": "%(levelprefix)s %(asctime)s - %(name)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
            "use_colors": True,
        },
        "access": {
            "()": "uvicorn.logging.AccessFormatter",
            "fmt": '%(levelprefix)s %(asctime)s - %(client_addr)s - "%(request_line)s" %(status_code)s',
            "datefmt": "%Y-%m-%d %H:%M:%S",
            "use_colors": True,
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
        "access": {
            "formatter": "access",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "uvicorn": {
            "handlers": ["default"],
            "level": LOG_LEVEL,
            "propagate": False,
        },
        "uvicorn.error": {
            "handlers": ["default"],
            "level": LOG_LEVEL,
            "propagate": False,
        },
        "uvicorn.access": {
            "handlers": ["access"],
            "level": LOG_LEVEL,
            "propagate": False,
        },
        "claudegate": {
            "handlers": ["default"],
            "level": LOG_LEVEL,
            "propagate": False,
        },
        # Silence noisy loggers unless debug mode
        "botocore": {
            "handlers": ["default"],
            "level": "WARNING" if LOG_LEVEL != "DEBUG" else LOG_LEVEL,
            "propagate": False,
        },
        "boto3": {
            "handlers": ["default"],
            "level": "WARNING" if LOG_LEVEL != "DEBUG" else LOG_LEVEL,
            "propagate": False,
        },
        "urllib3": {
            "handlers": ["default"],
            "level": "WARNING" if LOG_LEVEL != "DEBUG" else LOG_LEVEL,
            "propagate": False,
        },
    },
}

# Apply logging configuration
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("claudegate")

# Cross-region inference prefix (us, eu, apac) - required for on-demand throughput
# See: https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html
BEDROCK_REGION_PREFIX = os.environ.get("BEDROCK_REGION_PREFIX", DEFAULT_BEDROCK_REGION_PREFIX)

# Backend selection: "bedrock", "copilot", or comma-separated "copilot,bedrock" for fallback
_backends = [b.strip() for b in os.environ.get("BACKEND", "bedrock").lower().split(",") if b.strip()]
BACKEND_TYPE = _backends[0]  # primary backend
FALLBACK_BACKEND = _backends[1] if len(_backends) > 1 else ""  # optional fallback
FALLBACK_ON_ERRORS = {429, 500, 502, 503, 504}
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
COPILOT_TIMEOUT = int(os.environ.get("COPILOT_TIMEOUT", "300"))
