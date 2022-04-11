#include <opencv2/opencv.hpp>
#include <utility>
#include "ImageObject.h"


/**
 * Intiliser for Image Object
 * @param Image A loaded image via cv::imread
 */
ImageObject::ImageObject(cv::Mat Image) {
    image = std::move(Image);
}

/**
 * Short hand to access the current image instances height
 * @return The height of the image as an int
 */
int ImageObject::height() {
    return image.size().height;
}


/**
 * Short hand to access the current image instances width
 * @return The width of the image as an int
 */
int ImageObject::width() {
    return image.size().width;
}

/**
 * Show the current instance of Image
 * @param windowName The name of the window when shown, defaults to Window
 * @param delay The delay after which the window closes. If set to zero, which is the default, will wait until
 *      button pressed
 */
void ImageObject::showImage(const std::string &windowName, int delay) {
    cv::imshow(windowName, image);
    cv::waitKey(delay);
}
