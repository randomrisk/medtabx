"""Post-process perfusion maps."""
import numpy as np


def post_process_perfusion_map(perf_map, mask, map_type):
    """Post-process perfusion map with normalization.

    Args:
        perf_map: Perfusion map
        mask: Brain mask
        map_type: Map type ('CBF', 'CBV', 'MTT', 'TTP', 'TMAX')

    Returns:
        Processed perfusion map
    """
    # Remove outliers
    brain_values = perf_map[mask > 0]
    p1, p99 = np.percentile(brain_values, [1, 99])
    perf_map = np.clip(perf_map, p1, p99)

    # Normalize to 0-100 range for visualization
    if map_type.upper() in ['CBF', 'CBV']:
        brain_values = perf_map[mask > 0]
        if len(brain_values) > 0:
            mean_val = np.mean(brain_values)
            if mean_val > 0:
                perf_map = (perf_map / mean_val) * 50

    return perf_map
