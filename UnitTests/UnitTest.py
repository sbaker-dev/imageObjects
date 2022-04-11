from imageObjects.ImageObject import ImageObject
from imageObjects.Support import load_image

from pathlib import Path
import unittest


class MyTestCase(unittest.TestCase):

    @property
    def image_path(self):
        """Relative path to the image"""
        return Path(Path(__file__).parent, 'WastWater.jpg')

    def test_dimensions(self):
        """Test the image has not changed size by validating its known dimensions"""

        img = ImageObject(load_image(self.image_path))

        self.assertEqual(img.width, 574)
        self.assertEqual(img.height, 380)


if __name__ == '__main__':
    unittest.main()
