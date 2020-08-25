import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys


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

    def invert_image(self, new_image=False):
        """
        Invert the current image
        """
        inverted = cv2.bitwise_not(self.image)
        if new_image:
            return ImageObject(inverted)
        else:
            self.image = inverted

    @staticmethod
    def _morph_type(morph_type):
        """
        Use the morph_type key provided by the user to parse out the value to be used to set morphology via cv2
        """
        morph_values = {"erode": 0, "dilate": 1, "open": 2, "close": 3, "gradient": 4, "tophat": 5, "blackhat": 6}
        try:
            return morph_values[morph_type]
        except KeyError:
            sys.exit(f"ERROR: ImageObject.morphology\n"
                     f"Morphology type one of the following {list(morph_values.keys())} but found {morph_type}")

    @staticmethod
    def _kernel(kernel_type, kernel_vertical, kernel_horizontal):
        """
        Use kernel_type key provided by the user to parse out the kernel key to be used to construct the kernal of size
        dimensions equal to the kernel vertical and horizontal
        """
        kernel_values = {"rect": cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_vertical, kernel_horizontal)),
                         "cross": cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_vertical, kernel_horizontal)),
                         "ellipse": cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_vertical, kernel_horizontal))}
        try:
            return kernel_values[kernel_type]
        except KeyError:
            sys.exit("ERROR: ImageObjects.morphology\n"
                     f"Kernels can take on of the following {list(kernel_values.keys())} but found {kernel_type}")

    def morphology(self, morph_type, kernel_vertical, kernel_horizontal, kernel_type="rect", new_image=False):
        """
        This applies one of cv2's morph types to the image, or returns a new one if requested.

        morph_types
        ------------
        morph_types takes on of the following keywords: erode, dilate, open, close, gradient, tophat, blackhat

        kernel_types
        ------------
        kernel_type can take on the following keywords: rect, cross, ellipse. It defaults to rect
        """

        kernel = self._kernel(kernel_type, kernel_vertical, kernel_horizontal)
        if self._morph_type(morph_type) == 0:
            morphed = cv2.erode(self.image, kernel)
        elif self._morph_type(morph_type) == 1:
            morphed = cv2.dilate(self.image, kernel)
        elif 1 < self._morph_type(morph_type) < 7:
            morphed = cv2.morphologyEx(self.image, morph_type, kernel)
        else:
            sys.exit("CRITICAL ERROR: ImageObject.morphology\n"
                     "dict value return from self._morph_type() fell outside of operating range")

        if new_image:
            return ImageObject(morphed)
        else:
            self.image = morphed
