import numpy as np
from scipy import ndimage

def DSC(im1, im2):
    """
    dice coefficient 2nt/na + nb.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum


def vol_dif(im1, im2):
    """
    absolute difference in volume 
    """
    
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    return np.abs(im2.sum() - im1.sum()) / im1.sum()


def rtpf(im1, im2):
    """
    Regionwise True positive fraction
    """
    pass


def ftpf(im1, im2):
    """
    Regionwise True positive fraction
    """
    pass


    
