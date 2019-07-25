"""
U-Net
Common utility functions and classes.
Licensed under the MIT License (see LICENSE for details)
Written by Matthias Griebel
"""

import numpy as np
from scipy import ndimage
from scipy.spatial.distance import jaccard
from skimage.feature import peak_local_max
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import watershed

############################################################
#  Analyze regions and return labels
############################################################

def label_mask(mask, threshold=0.5, min_pixel=15, do_watershed=True, exclude_border=True):
    if mask.ndim == 3:
        mask = np.squeeze(mask, axis=2)

    # apply threshold to mask
    # bw = closing(mask > threshold, square(2))
    bw = (mask > threshold).astype(int)

    # label image regions
    label_image = label(bw)

    # Watershed: Separates objects in image by generate the markers
    # as local maxima of the distance to the background
    if do_watershed:
        distance = ndimage.distance_transform_edt(bw)
        # Minimum number of pixels separating peaks in a region of `2 * min_distance + 1`
        # (i.e. peaks are separated by at least `min_distance`)
        min_distance = int(np.ceil(np.sqrt(min_pixel / np.pi)))
        local_maxi = peak_local_max(distance, indices=False, exclude_border=False,
                                    min_distance=min_distance, labels=label_image)
        markers = label(local_maxi)
        label_image = watershed(-distance, markers, mask=bw)

    # remove artifacts connected to image border
    if exclude_border:
        label_image = clear_border(label_image)

    # remove areas < min pixel
    _ = [np.place(label_image, label_image == i, 0) for i in range(1, label_image.max()) if
         np.sum(label_image == i) < min_pixel]

    return (label_image)


############################################################
#  Compare masks using pixelwise Jaccard Similarity
############################################################

def jaccard_pixelwise(mask_a, mask_b, threshold=0.5):
    mask_a = (mask_a > threshold).astype(np.uint8)
    mask_b = (mask_b > threshold).astype(np.uint8)
    jac_dist = jaccard(mask_a.flatten(), mask_b.flatten())

    return (1 - jac_dist)

############################################################
#  Compare masks using ROI-wise Jaccard Similarity
############################################################

def jaccard_roiwise(mask_a, mask_b, threshold=0.5, min_roi_pixel=15, roi_threshold=0.5):
    labels_a = label_mask(mask_a, threshold=threshold, min_pixel=min_roi_pixel)
    labels_b = label_mask(mask_b, threshold=threshold, min_pixel=min_roi_pixel)
    label_stack = np.dstack((labels_a, labels_b))

    comb_cadidates = np.unique(label_stack.reshape(-1, label_stack.shape[2]), axis=0)
    # Remove Zero Entries
    comb_cadidates = comb_cadidates[np.prod(comb_cadidates, axis=1) > 0]

    jac = [1 - jaccard((labels_a == x[0]).astype(np.uint8).flatten(), (labels_b == x[1]).astype(np.uint8).flatten()) for
           x in comb_cadidates]
    matches = np.sum(np.array(jac) >= roi_threshold)
    union = (np.unique((labels_a)).size-1) + (np.unique((labels_b)).size-1) - matches

    return(matches/union)