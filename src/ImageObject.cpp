#include <opencv2/opencv.hpp>
#include <utility>
#include "ImageObject.h"


/**
 * Initializer for ImageObject
 * @param Image A loaded image via cv::imread
 */
ImageObject::ImageObject(cv::Mat Image) {
    image = std::move(Image);
}

/**
 * Return a new instance of ImageObject with the updated img, or update the current instance of ImageObject. To avoid
 * optional returns, both return a new instance of ImageObject
 * @param img A new Mat that was created by a method within ImageObject
 * @param newImage A bool of if the current image Mat should be replaced or not
 * @return An instance of ImageObject
 */
ImageObject ImageObject::updateOrExport(const cv::Mat& img, bool newImage) {

    if (newImage){
        return ImageObject(img);
    } else{
        image = img;
        return ImageObject(image);
    }

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
 * The number of channels in the image
 * @return Channel count
 */

int ImageObject::channels() {
    return image.channels();
}

/**
 * The total number of pixels in the image (width * height)
 * @return The total number of pixels
 */
int ImageObject::pixelTotal() {
    return width() * height();
}

/**
 * Total non zero elements of image. Most be gray, so creates a gray temp instance of image to valid
 * @return Total number of non zero images
 */
int ImageObject::nonZero() {
    ImageObject gray = changeToMono(true);
    return cv::countNonZero(gray.image);
}

/**
 * Validate if the image is empty by seeing if nonZero is equal to zero
 * @return True if empty, else False
 */

bool ImageObject::empty() {
    return nonZero() != 0;
}

/**
 * Extract a row from the Image
 * @param rowIndex A index of rows to extract
 * @return A given row from the image
 */
cv::Mat ImageObject::extractRow(int rowIndex) {
    return image.row(rowIndex);
}

/**
 * Extract a column from the Image
 * @param colIndex A index of columns to extract
 * @return A given column from the image
 */
cv::Mat ImageObject::extractCol(int colIndex) {
    return image.col(colIndex);
}

/**
 * Convert from BGR to RGB
 * @param newImage If you want a new instance or update the current reference
 * @return An instance of ImageObject
 */
ImageObject ImageObject::changeBgrToRgb(bool newImage) {
    cv::Mat output;
    cv::cvtColor(image, output, cv::COLOR_BGR2RGB);
    return updateOrExport(output, newImage);
}

/**
 * Convert from BGRA to BGR
 * @param newImage If you want a new instance or update the current reference
 * @return An instance of ImageObject
 */
ImageObject ImageObject::changeBgraToBgr(bool newImage) {
    cv::Mat output;
    cv::cvtColor(image, output, cv::COLOR_BGRA2BGR);
    return updateOrExport(output, newImage);
}

/**
 * Convert from BGR to BGRA
 * @param newImage If you want a new instance or update the current reference
 * @return An instance of ImageObject
 */
ImageObject ImageObject::changeBgrToBgra(bool newImage) {
    cv::Mat output;
    cv::cvtColor(image, output, cv::COLOR_BGR2BGRA);
    return updateOrExport(output, newImage);
}

/**
 * Convert from GRAY to BGR
 * @param newImage If you want a new instance or update the current reference
 * @return An instance of ImageObject
 */
ImageObject ImageObject::changeToColour(bool newImage) {
    cv::Mat output;
    cv::cvtColor(image, output, cv::COLOR_GRAY2BGR);
    return updateOrExport(output, newImage);
}

/**
 * Convert from BGR to GRAY
 * @param newImage If you want a new instance or update the current reference
 * @return An instance of ImageObject
 */
ImageObject ImageObject::changeToMono(bool newImage) {
    cv::Mat output;
    cv::cvtColor(image, output, cv::COLOR_BGR2GRAY);
    return updateOrExport(output, newImage);
}

/**
 * Invert the current image
 * @param newImage Return a new instance of ImageObject with the inverted image if True, otherwise updates current
 *  instance of image object
 * @return
 */
ImageObject ImageObject::invert(bool newImage) {
    cv::Mat output;
    cv::bitwise_not(image, output);
    return updateOrExport(output, newImage);
}


