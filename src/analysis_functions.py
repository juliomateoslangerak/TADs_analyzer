import numpy as np
import pandas as pd
from math import sqrt

from skimage.feature import blob_dog
from skimage.exposure import rescale_intensity
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label, regionprops_table
from skimage.segmentation import clear_border, watershed, relabel_sequential
from skimage.morphology import remove_small_objects
from porespy.metrics import regionprops_3D


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
        Rescaled image
    """
    image_dtype_max = (
        np.finfo(image.dtype).max
        if np.issubdtype(image.dtype, np.floating)
        else np.iinfo(image.dtype).max
    )

    # Use histogram to find mode for float arrays
    hist, bin_edges = np.histogram(image.ravel(), bins=bins)
    mode_bin_idx = np.argmax(hist)
    # Use the center of the bin with maximum count as the mode
    image_mode = (bin_edges[mode_bin_idx] + bin_edges[mode_bin_idx + 1]) / 2

    if out_range is None:
        out_range = (image_mode, image_dtype_max)

    return rescale_intensity(
        image=image, in_range=(image_mode, image_dtype_max), out_range=out_range
    )


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
        exclude_border=exclude_border,
    )


def process_channel(
    channel: np.ndarray,
    properties: tuple,
    subdomain_properties: tuple,
    voxel_size: tuple,
    sigma: float = None,
    min_volume: int = None,
    subdomain_min_volume: int = None,
    binarize: bool = True,
):
    # Preprocessing
    if sigma is None:
        filtered = channel
    else:
        filtered = gaussian(channel, sigma=sigma, preserve_range=True).astype("uint16")

    # Detecting Domains
    thresholded = filtered > threshold_otsu(filtered)
    domain_labels = label(thresholded)
    domain_labels = clear_border(domain_labels)
    if min_volume is not None:
        domain_labels = domain_labels > 0
        domain_labels = remove_small_objects(
            domain_labels, connectivity=domain_labels.ndim, min_size=min_volume
        )
        domain_labels = relabel_sequential(domain_labels.astype("uint8"))[0]
    if binarize:
        domain_labels = domain_labels > 0
        domain_labels = domain_labels.astype("uint8")
    domain_props_dict = regionprops_table(
        label_image=domain_labels, intensity_image=channel, properties=properties
    )
    domain_props_df = pd.DataFrame(domain_props_dict)
    pore_props_3d = regionprops_3D(domain_labels)
    domain_props_df["sphericity"] = 0
    domain_props_df["solidity"] = 0
    for lab in pore_props_3d:
        domain_props_df.loc[
            domain_props_df.label == lab.label, "sphericity"
        ] = lab.sphericity
        domain_props_df.loc[
            domain_props_df.label == lab.label, "solidity"
        ] = lab.solidity
    domain_props_df.insert(loc=0, column="roi_type", value="domain")

    # Detecting Subdomains
    subdomain_labels = watershed(np.invert(channel), mask=domain_labels)
    if subdomain_min_volume is not None:
        subdomain_labels = remove_small_objects(
            subdomain_labels, connectivity=subdomain_labels.ndim, min_size=min_volume
        )
        subdomain_labels = relabel_sequential(subdomain_labels, offset=2)[0]
    subdomain_props_dict = regionprops_table(
        label_image=subdomain_labels,
        intensity_image=channel,
        properties=subdomain_properties,
    )
    subdomain_props_df = pd.DataFrame(subdomain_props_dict)
    subdomain_props_df.insert(loc=0, column="roi_type", value="subdomain")

    # TODO: Add here nr of subdomains

    # Merging domain tables
    props_df = pd.concat([domain_props_df, subdomain_props_df], ignore_index=True)

    # Calculating some measurements
    props_df["volume"] = props_df["area"].apply(lambda a: a * np.prod(voxel_size))
    props_df["volume_units"] = "micron^3"

    return props_df, domain_labels, subdomain_labels


def process_image(
    image,
    domain_properties,
    subdomain_properties,
    voxel_size,
    sigma=None,
    min_volume=None,
    subdomain_min_volume=None,
    binarize=True,
):
    rois_df = pd.DataFrame()

    domain_labels = np.zeros_like(image, dtype="uint8")
    subdomain_labels = np.zeros_like(image, dtype="uint8")

    # this order (starting by channel number) is not defined by default
    for channel_index, channel in enumerate(image):
        (
            channel_props_df,
            channel_domain_labels,
            channel_subdomain_labels,
        ) = process_channel(
            channel=channel,
            properties=domain_properties,
            subdomain_properties=subdomain_properties,
            voxel_size=voxel_size,
            sigma=sigma,
            min_volume=min_volume,
            subdomain_min_volume=subdomain_min_volume,
            binarize=binarize,
        )

        domain_labels[channel_index] = channel_domain_labels
        subdomain_labels[channel_index] = channel_subdomain_labels

        channel_props_df.insert(loc=0, column="Channel ID", value=channel_index)

        rois_df = pd.concat([rois_df, channel_props_df], ignore_index=True)

    return rois_df, domain_labels, subdomain_labels
