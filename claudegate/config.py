"""Configuration constants and logging setup."""

import logging.config
import os
import ssl
import sys
from pathlib import Path

# Server defaults
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080

# Shared config directory
CONFIG_DIR = Path.home() / ".config" / "claudegate"
SERVER_URL_FILE = CONFIG_DIR / "server.json"
DEFAULT_LOG_LEVEL = "INFO"

# AWS defaults
DEFAULT_AWS_REGION = "us-west-2"
DEFAULT_BEDROCK_REGION_PREFIX = "us"
DEFAULT_READ_TIMEOUT = 300  # 5 minutes for slow models like Opus

# Logging
LOG_LEVEL = os.environ.get("CLAUDEGATE_LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()

# Respect NO_COLOR convention (https://no-color.org)
USE_COLORS = "NO_COLOR" not in os.environ

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
            "use_colors": USE_COLORS,
        },
        "access": {
            "()": "uvicorn.logging.AccessFormatter",
            "fmt": '%(levelprefix)s %(asctime)s - %(client_addr)s - "%(request_line)s" %(status_code)s',
            "datefmt": "%Y-%m-%d %H:%M:%S",
            "use_colors": USE_COLORS,
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
_backends = [b.strip() for b in os.environ.get("CLAUDEGATE_BACKEND", "copilot").lower().split(",") if b.strip()]
BACKEND_TYPE = _backends[0]  # primary backend
FALLBACK_BACKEND = _backends[1] if len(_backends) > 1 else ""  # optional fallback
FALLBACK_ON_ERRORS = {429, 500, 502, 503, 504}
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
COPILOT_TIMEOUT = int(os.environ.get("COPILOT_TIMEOUT", "300"))
COPILOT_MODELS_TTL = int(os.environ.get("COPILOT_MODELS_TTL", "300"))

# Retry on 429 before falling back (0 = no retries, go straight to fallback)
COPILOT_RETRY_MAX = int(os.environ.get("COPILOT_RETRY_MAX", "3"))
COPILOT_RETRY_BASE_DELAY = float(os.environ.get("COPILOT_RETRY_BASE_DELAY", "1.0"))

# Rate limiting: max requests per minute to Copilot (0 = unlimited)
COPILOT_MAX_RATE = int(os.environ.get("COPILOT_MAX_RATE", "15"))

# AI Framework backend — multi-environment (defaults are illustrative; set
# AI_FRAMEWORK_<ENV>_BASE_URL or override per deployment)
AI_FRAMEWORK_TIMEOUT = int(os.environ.get("AI_FRAMEWORK_TIMEOUT", "300"))

AI_FRAMEWORK_ENVIRONMENT_URLS: dict[str, str] = {
    "dev": "https://dev.ai-framework.example",
    "sandbox": "https://sandbox.ai-framework.example",
    "nprd": "https://nprd.ai-framework.example",
    "nprd-eu": "https://nprd-eu.ai-framework.example",
    "nprd-apac": "https://nprd-apac.ai-framework.example",
    "prod": "https://prod.ai-framework.example",
    "prod-eu": "https://prod-eu.ai-framework.example",
    "prod-apac": "https://prod-apac.ai-framework.example",
}
AI_FRAMEWORK_CHAT_PATH = "/ai-framework/ai-platform-serving/api/v1/chat/completions"

# Maps environment name to env var segment: "nprd-eu" -> "NPRD_EU"
AI_FRAMEWORK_ENV_KEY_MAP: dict[str, str] = {
    "dev": "DEV",
    "sandbox": "SANDBOX",
    "nprd": "NPRD",
    "nprd-eu": "NPRD_EU",
    "nprd-apac": "NPRD_APAC",
    "prod": "PROD",
    "prod-eu": "PROD_EU",
    "prod-apac": "PROD_APAC",
}


class AiFrameworkEnvConfig:
    """Configuration for a single AI Framework environment."""

    __slots__ = ("env_name", "service_credentials", "account_id", "base_url")

    def __init__(self, env_name: str, service_credentials: str, account_id: str, base_url: str):
        self.env_name = env_name
        self.service_credentials = service_credentials
        self.account_id = account_id
        self.base_url = base_url


def load_ai_framework_configs() -> dict[str, AiFrameworkEnvConfig]:
    """Scan env vars for AI_FRAMEWORK_<ENV>_SERVICE_CREDENTIALS and build per-env configs.

    Only environments with SERVICE_CREDENTIALS set are included.
    """
    configs: dict[str, AiFrameworkEnvConfig] = {}
    for env_name, key in AI_FRAMEWORK_ENV_KEY_MAP.items():
        creds = os.environ.get(f"AI_FRAMEWORK_{key}_SERVICE_CREDENTIALS", "")
        if not creds:
            continue
        account_id = os.environ.get(f"AI_FRAMEWORK_{key}_ACCOUNT_ID", "")
        base_url = os.environ.get(f"AI_FRAMEWORK_{key}_BASE_URL", "")
        if not base_url:
            base_url = AI_FRAMEWORK_ENVIRONMENT_URLS.get(env_name, "")
        if not base_url:
            logger.warning(f"AI Framework env '{env_name}': credentials set but no base URL available, skipping")
            continue
        configs[env_name] = AiFrameworkEnvConfig(env_name, creds, account_id, base_url)
    return configs


# Pre-flight context guard: reject requests estimated to exceed this fraction
# of the model's context limit. Set to 0 to disable. Default 0.90 leaves 10%
# headroom for tiktoken estimation inaccuracy.
CONTEXT_GUARD_THRESHOLD = float(os.environ.get("CONTEXT_GUARD_THRESHOLD", "0.90"))

# SSL context: prefer OS trust store (macOS Keychain, Windows CertStore) via truststore
# so corporate SSL inspection certs are trusted automatically. Fall back to httpx default
# (certifi CA bundle) if truststore is unavailable or crashes at runtime (e.g. CPython
# standalone builds on Linux where SSLObject._sslobj is None).
# Users on Linux behind corporate SSL inspection can set SSL_CERT_FILE or append their
# CA certs to certifi's cacert.pem.
try:
    import truststore  # noqa: E402

    SSL_CONTEXT = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    logger.debug("Using OS trust store via truststore (platform: %s)", sys.platform)
except Exception:
    SSL_CONTEXT = True  # httpx default: use certifi CA bundle
    logger.debug("truststore unavailable or incompatible, using certifi CA bundle (platform: %s)", sys.platform)
