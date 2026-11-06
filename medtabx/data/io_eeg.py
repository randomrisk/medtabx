"""
I/O module for EEG (Electroencephalography) data.

Supports multiple EEG file formats:
- EDF (European Data Format)
- FIF (Neuromag/Elekta format)
- BDF (BioSemi Data Format)
- SET (EEGLAB format)

Uses MNE-Python for reading EEG data.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path

try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    mne = None
    MNE_AVAILABLE = False


def load_eeg(
    filepath: str,
    preload: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load EEG data from file.

    Automatically detects format from extension.

    Args:
        filepath: Path to EEG file (.edf, .fif, .bdf, .set)
        preload: If True, load data into memory

    Returns:
        Tuple of (data_array, metadata_dict)
        - data_array: [n_channels, n_timepoints]
        - metadata: sampling rate, channel names, etc.

    Example:
        >>> data, meta = load_eeg('recording.edf')
        >>> print(data.shape)  # (64, 100000)
        >>> print(meta['sfreq'])  # 250.0 Hz
    """
    if not MNE_AVAILABLE:
        raise ImportError("MNE required. Install: pip install mne")

    filepath = Path(filepath)
    ext = filepath.suffix.lower()

    # Load based on format
    if ext == '.edf':
        raw = mne.io.read_raw_edf(str(filepath), preload=preload)
    elif ext == '.fif':
        raw = mne.io.read_raw_fif(str(filepath), preload=preload)
    elif ext == '.bdf':
        raw = mne.io.read_raw_bdf(str(filepath), preload=preload)
    elif ext == '.set':
        raw = mne.io.read_raw_eeglab(str(filepath), preload=preload)
    else:
        raise ValueError(f"Unsupported format: {ext}")

    # Get data
    data = raw.get_data()

    # Extract metadata
    metadata = {
        'sfreq': raw.info['sfreq'],
        'ch_names': raw.ch_names,
        'n_channels': len(raw.ch_names),
        'n_times': data.shape[1],
        'duration': data.shape[1] / raw.info['sfreq'],
    }

    return data, metadata
