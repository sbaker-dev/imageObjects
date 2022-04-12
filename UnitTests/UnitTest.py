from imageObjects.ImageObject import ImageObject
from imageObjects.Support import load_image

from pathlib import Path
import numpy as np
import unittest


class MyTestCase(unittest.TestCase):

    @property
    def image(self):
        """Relative path to the image"""
        return ImageObject(load_image(Path(Path(__file__).parent, 'WastWater.jpg')))

    def test_dimensions(self):
        """Test the image has not changed size by validating its known dimensions"""
        img = self.image
        self.assertEqual(img.width, 574)
        self.assertEqual(img.height, 380)
        self.assertEqual(img.pixel_total, 218120)
        self.assertEqual(img.channels, 3)

    def test_invert(self):
        """Test that the inversion of the image matrix's first row BGR total is equal to the known dimension"""
        inverted = self.image.invert(True)
        self.assertEqual(np.sum(inverted[0], axis=0).tolist(), [17379, 23935, 26712])


if __name__ == '__main__':
    unittest.main()
