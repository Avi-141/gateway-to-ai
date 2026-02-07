"""AWS Bedrock client management."""

import os

import boto3
from botocore.config import Config

from .config import DEFAULT_AWS_REGION, DEFAULT_READ_TIMEOUT

# Bedrock client - created via function to support credential refresh
_bedrock_client = None


def get_bedrock_client():
    """Get or create Bedrock client."""
    global _bedrock_client
    if _bedrock_client is None:
        # Configure longer timeout for slow models (Opus can take minutes)
        config = Config(
            read_timeout=int(os.environ.get("BEDROCK_READ_TIMEOUT", DEFAULT_READ_TIMEOUT)),
            retries={"max_attempts": 0},  # No retries - we handle errors explicitly
        )
        _bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name=os.environ.get("AWS_REGION", DEFAULT_AWS_REGION),
            config=config,
        )
    return _bedrock_client


def reset_bedrock_client():
    """Reset client to pick up new credentials after user re-authenticates."""
    global _bedrock_client
    _bedrock_client = None
    boto3.DEFAULT_SESSION = None
