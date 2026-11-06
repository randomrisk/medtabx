"""Generate brain masks from perfusion data."""
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_erosion, binary_dilation, binary_fill_holes
from skimage import segmentation, measure
from skimage.morphology import disk
import os


def generate_mask_ctp(volume, perf_path, bone_threshold=200, fast_marching_threshold=100):
    """Generate brain mask from CTP volume using fast marching.

    Args:
        volume: 3D CTP volume
        perf_path: Path to save mask
        bone_threshold: Threshold for bone detection
        fast_marching_threshold: Fast marching stopping value

    Returns:
        mask: Binary brain mask
        mask_path: Path to saved mask
    """
    print("Generating brain mask...")
    image = sitk.GetImageFromArray(volume)

    # Find seed point from largest bone component
    boneLabelMap = sitk.ConnectedComponent(image > bone_threshold)
    label_statistics = sitk.LabelShapeStatisticsImageFilter()
    label_statistics.Execute(boneLabelMap)

    label_list = [
        (i, image.TransformPhysicalPointToIndex(label_statistics.GetCentroid(i)),
         label_statistics.GetPhysicalSize(i))
        for i in label_statistics.GetLabels()
    ]
    seed = max(label_list, key=lambda x: x[2])[1]

    # Fast marching segmentation
    feature_image = sitk.GradientMagnitudeRecursiveGaussian(image, sigma=0.5)
    speed_image = sitk.BoundedReciprocal(feature_image)
    fm_filter = sitk.FastMarchingBaseImageFilter()
    fm_filter.SetTrialPoints([seed])
    fm_filter.SetStoppingValue(fast_marching_threshold)
    fm_image = fm_filter.Execute(speed_image)
    mask = fm_image < fast_marching_threshold

    mask_path = os.path.join(os.path.dirname(perf_path), 'brain_mask.nii.gz')
    sitk.WriteImage(mask, mask_path)
    mask = sitk.GetArrayFromImage(mask)

    return mask, mask_path


def generate_mask_mrp(volume, perf_path, brain_threshold=100):
    """Generate brain mask from MRP volume using active contours.

    Args:
        volume: 3D MRP volume
        perf_path: Path to save mask
        brain_threshold: Intensity threshold

    Returns:
        mask: Binary brain mask
        mask_path: Path to saved mask
    """
    print("Generating brain mask...")
    seed_mask = volume > brain_threshold
    mask = segmentation.morphological_chan_vese(volume, num_iter=300,
                                                 init_level_set=seed_mask).astype(float)

    # Remove skull fragments
    struct_elem = disk(4)[np.newaxis, :, :]
    mask = binary_erosion(mask, struct_elem)
    labeled_mask = measure.label(mask)
    regions = measure.regionprops(labeled_mask)

    if regions:
        max_area = max(region.area for region in regions)
        for region in regions:
            if region.area < max_area:
                mask[labeled_mask == region.label] = 0

    mask = binary_dilation(mask, struct_elem)
    mask = binary_fill_holes(mask).astype(np.uint8)

    mask_path = os.path.join(os.path.dirname(perf_path), 'brain_mask.nii.gz')
    mask_sitk = sitk.GetImageFromArray(mask)
    sitk.WriteImage(mask_sitk, mask_path)

    return mask, mask_path
