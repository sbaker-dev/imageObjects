from imageObjects import *

import unittest
import cv2


class ImageObjectsTest(unittest.TestCase):

    def test_load_image(self):
        """Test we have loaded the test image based on its known specifications"""
        image = ImageObject(cv2.imread("Page1.png"))

        self.assertEqual(image.height, 3508)
        self.assertEqual(image.width, 2481)
        self.assertEqual(image.channels, 3)
        self.assertEqual(image.count_colour_pixels(255), 24618849)
        self.assertEqual(image.count_colour_pixels(0), 1474758)

    def test_create_blank(self):
        """Test a blank image of width 50 height 100 is created of pure blank pixels"""
        image = create_blank(50, 100)
        self.assertEqual(image.width, 50)
        self.assertEqual(image.height, 100)
        self.assertEqual(image.count_colour_pixels(0), 5000)


if __name__ == '__main__':
    unittest.main()
