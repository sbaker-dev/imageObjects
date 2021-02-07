from vectorObjects.DefinedVectors import Vector2D
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

    @staticmethod
    def test_drawing_functions():
        """Test all the drawing functions of primitives from cv2"""
        image = create_blank(50, 100)

        image.draw_line(Vector2D(0, int(image.height / 3)), Vector2D(image.width, int(image.height / 3)),
                        (0, 255, 0), 1)
        image.draw_line((0, int(image.height / 2.5)), (image.width, int(image.height / 2.5)), (0, 0, 255), 1)
        image.draw_line([0, int(image.height / 2)], [image.width, int(image.height / 2)], (255, 0, 0), 1)

        image.draw_circle(Vector2D(int(image.width / 3), int(image.height / 3)), (255, 255, 0), 30)
        image.draw_circle((int(image.width / 2.5), int(image.height / 2.5)), (255, 0, 255), 30)
        image.draw_circle([int(image.width / 2), int(image.height / 2)], (0, 255, 255), 30)


if __name__ == '__main__':
    unittest.main()
