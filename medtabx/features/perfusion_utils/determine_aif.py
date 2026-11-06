"""Determine arterial input function (AIF)."""
import numpy as np
import pandas as pd
import cc3d
from scipy.optimize import curve_fit
from scipy.ndimage import binary_erosion, binary_dilation


def gamma_variate(t, t0, alpha, beta, amplitude):
    """Gamma variate function for AIF modeling."""
    t = np.array(t)
    t_shifted = np.maximum(0, t - t0)
    result = np.zeros_like(t_shifted)
    mask = t > t0
    result[mask] = amplitude * (t_shifted[mask]**alpha) * np.exp(-t_shifted[mask]/beta)
    return result


def fit_gamma_variate(time_index, curve, sigma=None):
    """Fit gamma variate to contrast curve."""
    properties, _ = curve_fit(
        gamma_variate, time_index, curve,
        bounds=([0, 0.1, 0.1, 0], [20, 8, 8, np.max(curve)*2]),
        sigma=sigma
    )
    return properties


def calculate_signal_smoothness(signal):
    """Calculate smoothness metric (lower = smoother)."""
    if len(signal) < 3:
        return float('inf')
    signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-8)
    second_derivative = np.diff(signal, n=2)
    return np.sum(second_derivative ** 2) / len(signal)


def determine_aif(ctc_volumes, time_index, mask, ttp, dilate_radius=5, erode_radius=2,
                  max_vol_thres=50, min_vol_thres=5, smoothness_threshold=0.04):
    """Identify arterial input function from contrast curves.

    Args:
        ctc_volumes: List of 3D contrast time curves
        time_index: Time indices
        mask: Brain mask
        ttp: Time-to-peak map
        dilate_radius: Dilation radius for morphological ops
        erode_radius: Erosion radius
        max_vol_thres: Max volume threshold
        min_vol_thres: Min volume threshold
        smoothness_threshold: Smoothness threshold

    Returns:
        aif_properties: Gamma variate parameters
        aif_mask: Binary AIF mask
    """
    # Erode mask to avoid skull
    kernel_size = mask.shape[2] // 15
    kernel = np.ones((1, kernel_size, kernel_size), dtype=bool)
    mask = binary_erosion(mask, kernel)

    ctc_volumes = np.stack(ctc_volumes, axis=0)
    auc_values = ctc_volumes.sum(0)[mask > 0]
    ttp_values = ttp[mask > 0]

    # Adaptive thresholding
    attempts = [(np.percentile(ttp_values, 100-p), np.percentile(auc_values, p))
                for p in range(95, 4, -10)]

    for attempt_ttp, attempt_auc in attempts:
        smoothness_attempts = [smoothness_threshold, smoothness_threshold * 2,
                               smoothness_threshold * 5, smoothness_threshold * 10]

        for current_smoothness_threshold in smoothness_attempts:
            aif_candidate = (ctc_volumes.sum(0) * mask > attempt_auc) * (ttp < attempt_ttp) * (ttp > 0)
            aif_candidate = binary_erosion(
                binary_dilation(aif_candidate, np.ones([1, dilate_radius, dilate_radius], bool)),
                np.ones([1, erode_radius, erode_radius], bool)
            )
            aif_candidate, nr_components = cc3d.connected_components(aif_candidate, return_N=True)

            if nr_components == 0:
                continue

            cands = []
            for idx in range(1, nr_components + 1):
                curve_candidates = aif_candidate == idx
                vol = curve_candidates.sum()
                curve = ctc_volumes[:, curve_candidates]
                curve_mean = curve.mean(axis=1)
                peak_difference = np.max(curve_mean) - np.mean(curve_mean[-3:])

                if vol > max_vol_thres or vol < min_vol_thres:
                    continue

                smoothness = calculate_signal_smoothness(curve_mean)
                if smoothness > current_smoothness_threshold:
                    continue

                try:
                    aif_properties = fit_gamma_variate(time_index, curve_mean)
                except:
                    continue

                fitting_error = np.sqrt(np.sum((curve - gamma_variate(time_index, *aif_properties)[:, np.newaxis]) ** 2, axis=1))
                cands.append({
                    'idx': idx,
                    'vol': vol,
                    'aif_properties': aif_properties,
                    'mean_error': fitting_error.mean(),
                    'peak_difference': peak_difference,
                })

            cands = pd.DataFrame(cands)
            if not cands.empty:
                cands['score'] = cands.vol * cands.peak_difference / cands.mean_error
                bestCand = np.argmax(cands.score)
                aifSegIdx = cands.iloc[bestCand].idx
                aif_properties = cands.iloc[bestCand].aif_properties
                aif_mask = (aif_candidate == aifSegIdx)
                return aif_properties, aif_mask

    raise ValueError("Could not find suitable AIF")
