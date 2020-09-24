import numpy as np
from imageObjects.ImageObject import ImageObject


def create_blank(width, height):
    """
    If the user has not defined anything, they can create a new image of zeros of size width-height
    :rtype: ImageObject
    """
    return ImageObject(np.zeros((height, width), dtype="float32"))

