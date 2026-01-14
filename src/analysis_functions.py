import numpy as np
from skimage.feature import blob_log, blob_dog
from skimage.exposure import rescale_intensity


def rescale_SIM(image, bins=256, out_range=None):
    """Rescale SIM image using modal intensity as the minimum and the datatype
    max as the maximum

    Parameters
    ----------
    image : np.array
        Input image array
    bins : int, optional
        Number of bins for histogram to find mode (default: 1000)
    out_range : str or tuple, optional
        Output range for rescaling. Can be 'dtype', 'uint16', or a tuple (min, max) (default: 'dtype')

    Returns
    -------
    np.ndarray
        Rescaled image as int16
    """
    image_dtype_max = np.finfo(image.dtype).max if np.issubdtype(image.dtype, np.floating) else np.iinfo(image.dtype).max

    # Use histogram to find mode for float arrays
    hist, bin_edges = np.histogram(image.ravel(), bins=bins)
    mode_bin_idx = np.argmax(hist)
    # Use the center of the bin with maximum count as the mode
    image_mode = (bin_edges[mode_bin_idx] + bin_edges[mode_bin_idx + 1]) / 2

    if out_range is None:
        out_range = (image_mode, image_dtype_max)

    return rescale_intensity(image=image, in_range=(image_mode, image_dtype_max), out_range=out_range)


def find_spots(
    image,
    min_sigma=5,
    max_sigma=20,
    threshold_rel=0.3,
    overlap=0.5,
    exclude_border=False,
):
    return blob_dog(
        image,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        threshold_rel=threshold_rel,
        overlap=overlap,
        exclude_border=exclude_border
    )