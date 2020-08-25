import cv2
import numpy as np
from matplotlib import pyplot as plt


class ImageObject:
    def __init__(self, image):
        self.image = image

    def show(self, window_name="Image"):
        """
        Show the image and wait for a button to be pressed to continue. Mainly designed for debugging processes
        """
        cv2.namedWindow(window_name, cv2.WINDOW_GUI_EXPANDED)
        cv2.imshow(window_name, self.image)
        cv2.waitKey()

    def notebook_show(self, title="Image"):
        """
        For jupyter we don't want to create a new image, and instead want to show an image via matplotlib.
        """
        plt.imshow(self.bgr_to_rgb(new_image=True).image)
        plt.title(title)
        plt.show()

    def bgr_to_rgb(self, new_image=False):
        """
        cv2 uses bgr rather than rgb, but this can be changed via this method
        """
        rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        if new_image:
            return ImageObject(rgb)
        else:
            self.image = rgb

    def colour_covert(self, new_image=False):
        """
        Convert image to colour
        """
        colour_image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        if new_image:
            return ImageObject(colour_image)
        else:
            self.image = colour_image

    def mono_convert(self, new_image):
        """
        Convert to a mono channel gray image
        """
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        if new_image:
            return ImageObject(gray)
        else:
            self.image = gray

    def invert_image(self, new_image=False):
        """
        Invert the current image
        """
        inverted = cv2.bitwise_not(self.image)
        if new_image:
            return ImageObject(inverted)
        else:
            self.image = inverted

    def change_a_colour(self, current_colour, new_colour, new_image=False):
        """
        Change all current pixels of one bgr colour to a different bgr colour
        """
        if new_image:
            image = self.image.copy()
            image[np.where((image == [current_colour]).all(axis=2))] = [new_colour]
            return ImageObject(image)
        else:
            self.image[np.where((self.image == [current_colour]).all(axis=2))] = [new_colour]






