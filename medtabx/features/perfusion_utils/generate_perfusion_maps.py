"""Generate perfusion maps from contrast time curves."""
import numpy as np
from scipy.linalg import toeplitz


def extract_ctc(volume_list, mask, image_type='ctp', echo_time=0.03, bolus_threshold=0.01):
    """Extract contrast time curve from 4D perfusion data.

    Args:
        volume_list: List of 3D volumes
        mask: Brain mask
        image_type: 'ctp' or 'mrp'
        echo_time: Echo time for MRP
        bolus_threshold: Threshold for bolus detection

    Returns:
        volume_list: Contrast signal volumes
        s0_index: Baseline index
        norm_list_of_means: Normalized mean intensities
    """
    if image_type == 'mrp':
        global_max = max(np.max(vol) for vol in volume_list)
        global_min = min(np.min(vol) for vol in volume_list)
        volume_list = [global_max + global_min - i for i in volume_list]

    # Calculate mean signal in brain
    list_of_means = [np.mean(vol[mask > 0]) for vol in volume_list]
    norm_list_of_means = [m / list_of_means[0] for m in list_of_means]

    # Find bolus arrival
    s0_index = 0
    for i in range(1, len(norm_list_of_means)):
        if image_type == 'ctp':
            if norm_list_of_means[i] > norm_list_of_means[i-1] * (1 + bolus_threshold):
                s0_index = i
                break
        else:
            if norm_list_of_means[i] < norm_list_of_means[i-1] * (1 - bolus_threshold):
                s0_index = i
                break

    s0_index = max(0, s0_index - 1)

    # Calculate baseline
    volumes = np.stack(volume_list, axis=0)
    s0 = volumes[:s0_index+1].mean(axis=0)

    # Extract contrast signal
    if image_type == 'ctp':
        volumes = volumes - s0[np.newaxis, :, :, :]
        volumes = np.maximum(volumes, 0)
    else:
        epsilon = 1e-8
        s0 = np.where(s0 == 0, epsilon, s0)
        ratio = volumes / (s0[np.newaxis, :, :, :])
        ratio = np.where(ratio <= 0, epsilon, ratio)
        volumes = (1 / echo_time) * np.log(ratio)

    volumes = volumes * mask[np.newaxis, :, :, :]
    volume_list = [volumes[i] for i in range(volumes.shape[0])]

    return volume_list, s0_index, norm_list_of_means


def generate_ttp(ctc_volumes, time_index, s0_index, mask, outside_value=-1):
    """Generate time-to-peak map.

    Args:
        ctc_volumes: List of 3D contrast volumes
        time_index: Time indices
        s0_index: Baseline index
        mask: Brain mask
        outside_value: Value for outside mask

    Returns:
        ttp: Time-to-peak map
    """
    ctc_volumes = np.stack(ctc_volumes, axis=0)
    time_index = np.array(time_index)
    time_index = time_index[s0_index:] - time_index[s0_index]
    ctc_volumes = ctc_volumes[s0_index:, :, :, :]
    ttp = time_index[ctc_volumes.argmax(axis=0)]
    ttp[mask == 0] = outside_value
    return ttp


def generate_perfusion_maps(ctc_volumes, time_index, mask, aif_properties, method='bcSVD1',
                            SVD_truncation_threshold=0.1, outside_value=-1, rho=1.05, hcf=0.73):
    """Generate perfusion maps via deconvolution.

    Args:
        ctc_volumes: List of 3D contrast volumes
        time_index: Time indices
        mask: Brain mask
        aif_properties: AIF gamma variate parameters
        method: Deconvolution method ('SVD', 'bcSVD1', 'bcSVD2')
        SVD_truncation_threshold: SVD threshold
        outside_value: Value for outside mask
        rho: Tissue density (g/ml)
        hcf: Hematocrit correction factor

    Returns:
        mtt, cbv, cbf, tmax: Perfusion maps
    """
    from .determine_aif import gamma_variate

    if method not in ['SVD', 'bcSVD1', 'bcSVD2']:
        raise ValueError("Invalid method")

    aif = gamma_variate(time_index, *aif_properties)
    deltaT = np.mean(np.diff(time_index))
    nr_timepoints = len(time_index)

    # Build convolution matrix
    if method == 'SVD':
        G = toeplitz(aif, np.zeros(nr_timepoints))
        ctc_volumes_pad = ctc_volumes
    elif method == 'bcSVD1':
        colG = np.zeros(2 * nr_timepoints)
        colG[0] = aif[0]
        colG[nr_timepoints - 1] = (aif[nr_timepoints - 2] + 4 * aif[nr_timepoints - 1]) / 6
        colG[nr_timepoints] = aif[nr_timepoints - 1] / 6
        for k in range(1, nr_timepoints - 1):
            colG[k] = (aif[k - 1] + 4 * aif[k] + aif[k + 1]) / 6
        rowG = np.zeros(2 * nr_timepoints)
        rowG[0] = colG[0]
        for k in range(1, 2 * nr_timepoints):
            rowG[k] = colG[2 * nr_timepoints - k]
        G = toeplitz(colG, rowG)
        ctc_volumes_pad = np.pad(ctc_volumes, [(0, len(ctc_volumes)),] + [(0, 0)] * 3)
    else:  # bcSVD2
        cmat = np.zeros([nr_timepoints, nr_timepoints])
        B = np.zeros([nr_timepoints, nr_timepoints])
        for i in range(nr_timepoints):
            for j in range(nr_timepoints):
                if i == j:
                    cmat[i, j] = aif[0]
                elif i > j:
                    cmat[i, j] = aif[(i - j)]
                else:
                    cmat[i, j] = aif[nr_timepoints - (j - i)]
                B[i, j] = aif[abs(i - j)]
        G = np.block([[cmat, B], [B, cmat]])
        ctc_volumes_pad = np.pad(ctc_volumes, [(0, len(ctc_volumes)),] + [(0, 0)] * 3)

    # SVD deconvolution
    U, S, V = np.linalg.svd(G * deltaT)
    threshold = SVD_truncation_threshold * S[0]
    filteredS = 1.0 / (S + 1e-5)
    filteredS[S < threshold] = 0
    Ginv = V.T @ np.diag(filteredS) @ U.T
    R = np.abs(np.einsum('ab, bcde->acde', Ginv, ctc_volumes_pad))
    R = R[:nr_timepoints]

    # Calculate perfusion parameters
    cbf = hcf / rho * R.max(axis=0) * 60 * 100
    cbv = hcf / rho * R.sum(axis=0) * 100
    mtt = np.divide(cbv, cbf, out=np.zeros_like(cbv), where=cbf!=0) * 60
    time_index = np.array(time_index, dtype=int)
    tmax = time_index[R.argmax(axis=0)].astype(np.float64)

    tmax[mask == 0] = outside_value
    cbv[mask == 0] = outside_value
    cbf[mask == 0] = outside_value
    mtt[mask == 0] = outside_value

    return mtt, cbv, cbf, tmax
