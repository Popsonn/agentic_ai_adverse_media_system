# adverse_media_system/utils/text_processing.py

import json
from urllib.parse import urlparse

def safe_json_parse(json_str: str, default=None):
    """Safely parse a JSON string, returning a default value on failure."""
    if default is None:
        default = {}
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default

def extract_domain(url: str) -> str:
    """Extracts the network location (domain) from a URL."""
    if not url:
        return ""
    try:
        return urlparse(url).netloc.lower()
    except:
        return ""