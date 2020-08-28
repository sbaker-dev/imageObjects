from imageObjects.ImageObject import ImageObject
from imageObjects import common as cf
import numpy as np
import cv2
import sys


class LineObject:
    def __init__(self, image, target_colour, min_length, mark_colour=180, block_out=None, warnings=True):
        if isinstance(image, ImageObject):
            self.img = image.shape_mask(target_colour, target_colour, new_image=True)
            self.lines = None

            self._mark = mark_colour
            self._mark_bgr = (mark_colour, mark_colour, mark_colour)
            self._block_out = block_out
            self._len_min = min_length

            self._WHITE = 255
            self._WHITE_BGR = (255, 255, 255)
            self._BLACK = 0
            self._BLACK_BGR = (0, 0, 0)

            if warnings and (cv2.countNonZero(self.img.image) / self.img.pixel_total() > 0.5):
                print("Warning: The number of pixels you are running is more than 50% of the image and may cause "
                      "significant slowdown.\nYou can turn this warning off with warnings=False")

        else:
            sys.exit("LineObject expects an ImageObject")

    def find_horizontal_lines(self, adjacent_pixels=None, fill_gaps_max_length=None):
        """
        This will look for horizontal lines that meet the minimum width requirement

        :param adjacent_pixels: If specified, this number of adjacent pixels will also be selected from each line found
        :type adjacent_pixels: int | None

        :param fill_gaps_max_length:  If specified, this will try to find and fill gaps between lines that might occur
            on damaged or old images. The integer passed is the maximum gap you will permit, the min is taken from
            min_length passed to the class object.
        :type fill_gaps_max_length int | None

        """
        # For each row of the image, find how many of the current row are not zero, grouping them by adjacency. If the
        # adjacent groups meet the minimum length requirement, draw them. Once all rows are complete, isolate the lines
        for i in range(self.img.height):
            indexes_list = cf.group_adjacent(cf.flatten(np.asarray((self.img.image[i, :]).nonzero())))
            [self._draw_horizontal_lines(i, x_indexes) for x_indexes in indexes_list]
        self.img.shape_mask(self.mark_bgr, self.mark_bgr)

        if adjacent_pixels:
            [c.draw_contour(self.img, self._WHITE_BGR, adjacent_pixels) for c in self.img.find_contours("external")]

        self.img.show()

    def _draw_horizontal_lines(self, i, x_indexes):
        """
        Draws a horizontal line on an image as long, as it meets the minimum length requirement, via pixel indexes.

        For the horizontal draw lines our index i is our row index and our in line ii is the column index.
        """
        return [self.img.change_pixel_colour(i, ii, self._mark) for ii in x_indexes if len(x_indexes) > self._len_min]
