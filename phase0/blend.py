"""Shim: the core KV code now lives in the `tread` package (tread/blend.py).

Phase 0 scripts run as `python phase0/<script>.py`, so put the repo root on sys.path and re-export.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tread.blend import (  # noqa: E402,F401
    _layer_subset, _qkv, assemble, blend, chunk_kv, layer1_deviation, prefill_kv, rotate_keys, to_cache,
)
