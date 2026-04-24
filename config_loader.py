"""Centralised config loader.  Import `CFG` anywhere."""
import yaml, os
from pathlib import Path

_CFG_PATH = Path(__file__).parent / "config.yaml"

def _load():
    with open(_CFG_PATH) as f:
        return yaml.safe_load(f)

CFG = _load()

def reload():
    global CFG
    CFG = _load()
    return CFG
