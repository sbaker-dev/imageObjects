from imageObjects.ContourObject import ContourObject
from matplotlib import pyplot as plt
import numpy as np
import cv2
import sys


class ImageObject:
    def __init__(self, image):
        self.image = image

    def __repr__(self):
        """
        Return for debug
        """
        return f"{self.image.shape}"

    def __str__(self):
        """
        Clearer message of what the current instance contains
        """
        if self.channels < 3:
            channels_type = "Mono"
        else:
            channels_type = "Colour"

        return f"Height: {self.height}\nWidth: {self.width}\nType: {channels_type}\nChannels: {self.channels}"

    @property
    def height(self):
        """
        The height of the image in pixels
        """
        return self.image.shape[0]

    @property
    def width(self):
        """
        The width of the image in pixels
        """
        return self.image.shape[1]

    @property
    def channels(self):
        """
        The number of channels of colour data that exist in the image
        """
        if len(self.image.shape) < 3:
            return 1
        else:
            return self.image.shape[2]

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
                     f"Kernels can take one of the following {list(kernel_values.keys())} but found {kernel_type}")

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

    @staticmethod
    def _key_return(method_name, dict_name, dict_of_values, key):
        try:
            return dict_of_values[key]
        except KeyError:
            raise KeyError(f"{method_name}s {dict_name} only takes {list(dict_of_values.keys())} but found {key}")

    @staticmethod
    def _retrieval_mode(retrieval_mode):
        """
        The retrieval mode determines the hierarchy of contours, explain in full in the cv2 docs:

        https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours
        /py_contours_hierarchy/py_contours_hierarchy.html

        retrieval_mode
        -------------
        retrieval_mode can take on of the following values: external, list, ccomp, tree, floodfill
        """
        retrieval_values = {"external": 0, "list": 1, "ccomp": 2, "tree": 3, "floodfill": 4}
        try:
            return retrieval_values[retrieval_mode]
        except KeyError:
            sys.exit("ERROR: ImageObjects.find_contours\n"
                     f"Retrieval takes one of the following {list(retrieval_values.keys())} but found {retrieval_mode}")

    def _create_temp_image(self, colour=True):
        """
        Sometimes we need to create a temporary image, when certain operations require a mono or three channel or more
        image and the current image is not of that type. This creates a colour image by default, but can also create
        mono image, of the current image. If the current image actually meets those specifications, then it is just
        duplicated.
        """
        if colour and self.channels < 3:
            image = self.colour_covert(new_image=True).image
        elif not colour and self.channels > 2:
            image = self.mono_convert(new_image=True).image
        else:
            image = self.image.copy()

        return image

    def find_contours(self, retrieval_mode, simple_method=True, hierarchy_return=False):
        """
        Find contours within the current image. Since find contours only works on mono channel images, if the current
        image is a colour image a new image is create that will not change the one in memory so it is not necessary to
        change the image to gray manually.

        retrieval_mode
        ---------------
        retrieval_mode can take on of the following values: external, list, ccomp, tree, floodfill

        simple_method
        --------------
        Simple methods will only use the extremes, so for a straight line it will store the first and last point. If you
        turn this off it will keep all the points on the line but this can lead to a very large over head so it is not
        recommend unless you have a specific need for all those points.

        hierarchy_return
        ----------------
        If you don't want to return the hierarchy you can leave this as default, otherwise set it to true
        """

        # Setup of extraction methods
        retrieval = self._retrieval_mode(retrieval_mode)
        if simple_method:
            approx = cv2.CHAIN_APPROX_SIMPLE
        else:
            approx = cv2.CHAIN_APPROX_NONE

        # Look for contours base on the setup
        contours, hierarchy = cv2.findContours(self._create_temp_image(colour=False), retrieval, approx)

        # If we find any contours, return the contours and the hierarchy if requested
        if len(contours) > 0:
            if hierarchy_return:
                return [ContourObject(cnt) for cnt in contours], hierarchy
            else:
                return [ContourObject(cnt) for cnt in contours]
        else:
            if hierarchy_return:
                return None, None
            else:
                return None

    def alpha_mask(self, new_image=False):
        """
        Takes a 4 channel image and returns the alpha channel as a mask
        """
        _, mask = cv2.threshold(self.image[:, :, 3], 0, 255, cv2.THRESH_BINARY)
        if new_image:
            return ImageObject(mask)
        else:
            self.image = new_image

    def blank_like(self, new_image=False):
        """
        Create a blank image of the same dimensions as the image
        """
        blank = np.zeros_like(self.image)
        if new_image:
            return ImageObject(blank)
        else:
            self.image = blank

    def extend_bounds(self, uniform=True, size=1, top=1, bottom=1, left=1, right=1, colour=(0, 0, 0), new_image=False):
        """
        This will extend the bounds of the image, if uniform is selected then you only need to adjust size and all
        borders will extend by that amount. Otherwise you can find tweak the size increase through the top, bottom,
        left, right parameters.
        """
        if uniform:
            output_border = cv2.copyMakeBorder(self.image, size, size, size, size, cv2.BORDER_CONSTANT, value=colour)
        else:
            output_border = cv2.copyMakeBorder(self.image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=colour)

        if new_image:
            return ImageObject(output_border)
        else:
            self.image = output_border

    def shape_mask(self, lower_thresh, upper_thresh, new_image=False):
        """
        Isolate shapes within a given bgr range.
        """
        shape_mask = cv2.inRange(self._create_temp_image(), lower_thresh, upper_thresh)
        if new_image:
            return ImageObject(shape_mask)
        else:
            self.image = shape_mask

    def mask_image(self, mask, new_image=False):
        """
        This will use another image as a mask for this image. Can be an ImageObject or any cv2 compatible image.
        """
        if isinstance(mask, ImageObject):
            masked = cv2.bitwise_and(self.image, self.image, mask.image)
        else:
            masked = cv2.bitwise_and(self.image, self.image, mask)

        if new_image:
            return ImageObject(masked)
        else:
            self.image = masked

    def binary_threshold(self, binary_threshold, binary_mode="binary", binary_max=255, new_image=False):
        """
        Create or push the image to a binary black on white image based on a threshold of mono pixel values.
        """
        binary_values = {"binary": 0, "binary_inv": 1, "trunc": 2, "to_zero": 3, "to_zero_inv": 4}
        binary_mode = self._key_return("binary_threshold", "binary_mode", binary_values, binary_mode)

        _, threshold_image = cv2.threshold(self.image, binary_threshold, binary_max, binary_mode)

        if new_image:
            return ImageObject(threshold_image)
        else:
            self.image = threshold_image

    def calculate_alpha_beta(self, clip_hist_percent=1):
        """
        This calculates the current alpha and beta values from an image

        From https://stackoverflow.com/questions/56905592/automatic-contrast-and-brightness-adjustment-of-a-color-photo-
        of-a-sheet-of-paper
        """
        gray = self._create_temp_image(colour=False)

        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_size = len(hist)

        # Calculate cumulative distribution from the histogram
        previous = float(hist[0])
        accumulator = [previous]
        for index in range(1, hist_size):
            previous = previous + float(hist[index])
            accumulator.append(previous)

        # Locate points to clip
        maximum = accumulator[-1]
        clip_hist_percent *= (maximum / 100.0)
        clip_hist_percent /= 2.0

        # Locate left cut
        minimum_gray = 0
        while accumulator[minimum_gray] < clip_hist_percent:
            minimum_gray += 1

        # Locate right cut
        maximum_gray = hist_size - 1
        while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
            maximum_gray -= 1

        # Calculate alpha and beta values
        alpha = 255 / (maximum_gray - minimum_gray)
        beta = -minimum_gray * alpha

        return alpha, beta
