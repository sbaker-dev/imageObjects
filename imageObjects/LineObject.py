from imageObjects.ImageObject import ImageObject
from imageObjects import common as cf
import numpy as np
import cv2
import sys


class LineObject:
    def __init__(self, image, target_colour, min_length, mark_colour=180, block_colour=255, warnings=True):
        if isinstance(image, ImageObject):
            # Isolate the pixels we need to work with, and check the percentage of the master to stop massive slowdowns
            self.img = image.shape_mask(target_colour, target_colour, new_image=True)
            self.mark = mark_colour
            self.mark_bgr = (mark_colour, mark_colour, mark_colour)
            self.block = block_colour
            self.length_min = min_length

            if warnings and (cv2.countNonZero(self.img.image) / self.img.pixel_total() > 0.5):
                print("Warning: The number of pixels you are running is more than 50% of the image and may cause "
                      "significant slowdown.\nYou can turn this warning off with warnings=False")

        else:
            sys.exit("LineObject expects an ImageObject")

    def find_horizontal_lines(self, adjacent_runs=None):
        """
        This will look for horizontal lines that meet the minimum width requirement
        """

        # For each row of the image, find how many of the current row are not zero, grouping them by adjacency. If the
        # adjacent groups meet the minimum length requirement, draw them. Once all rows are complete, isolate the lines
        for i in range(self.img.height):
            indexes_list = cf.group_adjacent(cf.flatten(np.asarray((self.img.image[i, :]).nonzero())))
            [self._draw_horizontal_lines(i, x_indexes) for x_indexes in indexes_list]
        self.img.shape_mask(self.mark_bgr, self.mark_bgr)

        if adjacent_runs:
            self._fill_adjacent(adjacent_runs)

    def _draw_horizontal_lines(self, i, x_indexes):
        """
        Draws a horizontal line on an image as long, as it meets the minimum length requirement, via pixel indexes.

        For the horizontal draw lines our index i is our row index and our in line ii is the column index.
        """
        return [self.img.change_pixel_colour(i, ii, self.mark) for ii in x_indexes if len(x_indexes) > self.length_min]

    def _fill_adjacent(self, runs):
        """
        This is designed to isolate all the pixels that where found, and then get the adjacent pixels. From there we
        then want fill all the pixels that are within the bounds of the image.
        """
        x, y = self.img.image.nonzero()
        target_pixels = [[xv, yv] for xv, yv in zip(x, y)]

        # For each run, we extend the bounds of the pixels we isolate by 1
        for x, y in target_pixels:
            for i in range(1, runs + 1):
                adjacent_four = [(x + i, y), (x - i, y), (x, y - i), (x, y + i)]
                for adjacent_x, adjacent_y in adjacent_four:
                    try:
                        self.img.change_pixel_colour(adjacent_x, adjacent_y, self.mark)
                    except IndexError:
                        pass

        self.img.shape_mask(self.mark_bgr, self.mark_bgr)

