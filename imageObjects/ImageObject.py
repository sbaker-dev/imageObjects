import cv2


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

    def invert_image(self, new_image=False):
        """
        Invert the current image
        """
        inverted = cv2.bitwise_not(self.image)
        if new_image:
            return ImageObject(inverted)
        else:
            self.image = inverted


