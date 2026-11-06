"""
I/O module for ECG (Electrocardiography) data.

Supports:
- WFDB format (PhysioNet)
- EDF format
- CSV format
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import warnings
from pathlib import Path

# Delay wfdb import to avoid pandas compatibility issues at module load time
WFDB_AVAILABLE = None  # Will be checked when needed

def _check_wfdb():
    """Check if wfdb is available and can be imported."""
    global WFDB_AVAILABLE
    if WFDB_AVAILABLE is None:
        try:
            import wfdb
            WFDB_AVAILABLE = True
        except (ImportError, TypeError) as e:
            WFDB_AVAILABLE = False
            if isinstance(e, ImportError):
                warnings.warn("wfdb not installed. Install: pip install wfdb")
            else:
                warnings.warn(f"wfdb has compatibility issues: {e}")
    return WFDB_AVAILABLE


def _parse_gain(gain_token: str) -> float:
    """Parse WFDB gain token like '1000.0(0)/mV'."""
    base = gain_token.split("/")[0]
    gain_str = base.split("(")[0]
    try:
        gain = float(gain_str)
        return gain if gain != 0 else 1.0
    except ValueError:
        return 1.0


def _load_wfdb_fallback(record_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load WFDB record from .hea/.dat for common 16-bit interleaved format."""
    record = Path(record_path)
    hea_path = record.with_suffix(".hea")
    dat_path = record.with_suffix(".dat")

    if not hea_path.exists() or not dat_path.exists():
        raise FileNotFoundError(
            f"WFDB record files not found for base path: {record_path}"
        )

    lines = hea_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Empty WFDB header: {hea_path}")

    first = lines[0].split()
    # Expected: record_name n_sig fs sig_len
    if len(first) < 4:
        raise ValueError(f"Invalid WFDB header first line: {lines[0]}")

    n_sig = int(first[1])
    fs = float(first[2])
    sig_len = int(first[3])

    sig_lines = lines[1 : 1 + n_sig]
    if len(sig_lines) != n_sig:
        raise ValueError(f"WFDB header expected {n_sig} signal lines, got {len(sig_lines)}")

    sig_name = []
    units = []
    gains = []
    for line in sig_lines:
        parts = line.split()
        if len(parts) < 9:
            raise ValueError(f"Invalid WFDB signal line: {line}")
        # parts[1] should be format, support only 16-bit in this fallback.
        fmt = parts[1]
        if fmt != "16":
            raise ValueError(
                f"Fallback WFDB parser supports only format 16, got '{fmt}'"
            )
        gains.append(_parse_gain(parts[2]))
        units.append(parts[2].split("/")[-1] if "/" in parts[2] else "adu")
        sig_name.append(parts[8])

    raw = np.fromfile(dat_path, dtype="<i2")
    expected = sig_len * n_sig
    if raw.size < expected:
        raise ValueError(
            f"WFDB data size mismatch: expected at least {expected} samples, got {raw.size}"
        )
    if raw.size > expected:
        raw = raw[:expected]

    # WFDB dat for format 16 is channel-interleaved by sample index.
    data = raw.reshape(sig_len, n_sig).astype(np.float32)
    gain_arr = np.asarray(gains, dtype=np.float32)
    p_signal = data / gain_arr

    metadata = {
        "fs": fs,
        "sig_len": sig_len,
        "sig_name": sig_name,
        "units": units,
        "loader_impl": "wfdb_fallback",
    }
    return p_signal, metadata


def load_ecg_wfdb(
    record_path: str,
    channels: Optional[list] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load ECG from WFDB format.

    Args:
        record_path: Path to WFDB record (without extension)
        channels: List of channel indices to load

    Returns:
        Tuple of (signal_array, metadata)

    Example:
        >>> ecg, meta = load_ecg_wfdb('data/100')
        >>> print(ecg.shape)  # (650000, 2)
    """
    if _check_wfdb():
        import wfdb

        record = wfdb.rdrecord(record_path, channels=channels)
        metadata = {
            'fs': record.fs,
            'sig_len': record.sig_len,
            'sig_name': record.sig_name,
            'units': record.units,
            'loader_impl': 'wfdb',
        }
        return record.p_signal, metadata

    warnings.warn(
        "wfdb import unavailable or incompatible; using built-in WFDB fallback parser"
    )
    data, metadata = _load_wfdb_fallback(record_path)
    if channels is not None:
        data = data[:, channels]
        metadata['sig_name'] = [metadata['sig_name'][idx] for idx in channels]
        metadata['units'] = [metadata['units'][idx] for idx in channels]
    return data, metadata
