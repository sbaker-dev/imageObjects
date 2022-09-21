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
 * @return An instance of ImageObject
 */
ImageObject ImageObject::invert(bool newImage) {
    cv::Mat output;
    cv::bitwise_not(image, output);
    return updateOrExport(output, newImage);
}

/**
 * Isolate a vector of contours, with a single contour being a vector of cv::Points, based on a given retrieval mode and
 * approx method
 * @param retrieval_mode The type of contour retrieval, takes one of the following values: external, list, ccomp, tree,
 *  floodfill
 * @param simple_approx: Simple will only use the extremes that are needed to define a shape, where as if it is turned
 *  off, all points are kept
 * @return A Vector of Vectors of Points.
 */
std::vector<std::vector<cv::Point>> ImageObject::extractContours(const std::string& retrieval_mode, bool simple_approx) {
    cv::Mat extractImage;
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;

    // TODO: We probably want a static method for creating temp images of a given channel depth
    if (channels() != 1){
        extractImage = changeToMono(true).image;
    }
    else{
        extractImage = image.clone();
    }

    // Determine if we are using CHAIN_APPROX_SIMPLE or CHAIN_APPROX_NONE
    int approx;
    if (simple_approx){
        approx = 2;
    } else{
        approx = 1;
    }

    // Remap string to int
    std::map<std::string, int> retrieval;
    retrieval["external"] = 0;
    retrieval["list"] = 1;
    retrieval["ccomp"] = 2;
    retrieval["tree"] = 3;
    retrieval["floodfill"] = 4;

    //
    if (retrieval.count(retrieval_mode) == 0){
        std::cout << "We failed to find a given retrieval mode for " << retrieval_mode << std::endl;
    } else {
        int mode = retrieval[retrieval_mode];
        cv::findContours(extractImage, contours, hierarchy, mode, approx);
    }

    // TODO: We need to allow for returning the hierarchy?
    return contours;
}

/**
 * Draw a given contour on the image
 * @param contours Extract contours from an image
 * @param index The index position you want to draw, -1 if you want to draw all of them
 * @param thickness The thickness of the contour, -1 if you want to fill
 * @param newImage If you want to draw on current image or return a new instance
 * @return An instance of ImageObject
 */
ImageObject ImageObject::drawContour(const std::vector<std::vector<cv::Point>>& contours, int index,
        const cv::Scalar& colour, int thickness, bool newImage) {

    cv::Mat output = image.clone();
    cv::drawContours(output, contours, index, colour, thickness);
    return updateOrExport(output, newImage);
}

/**
 * Threshold Binary a given image
 * @param binaryThreshold The level of thresh holding
 * @param binaryMode Mode of thresh holding
 * @param binaryMax maximum cap value for thresh holding
 * @param newImage new image or not
 * @return
 */
ImageObject ImageObject::thresholdBinary(int binaryThreshold, const std::string &binaryMode, int binaryMax,
                                         bool newImage) {

    cv::Mat output;

    // Remap string to int
    std::map<std::string, int> retrieval;
    retrieval["binary"] = 0;
    retrieval["binary_inv"] = 1;
    retrieval["trunc"] = 2;
    retrieval["to_zero"] = 3;
    retrieval["to_zero_inv"] = 4;

    // Binary threshold the image
    if (retrieval.count(binaryMode) == 0){
        std::cout << "We failed to find a given retrieval mode for " << binaryMode << std::endl;
    } else {
        int mode = retrieval[binaryMode];
        cv::threshold(image, output, binaryThreshold, binaryMax, mode);
    }
    return updateOrExport(output, newImage);
}

/**
 * This will apply by default a gaussian adaptive threshold using the binary method
 * @param assignmentValue Pixels threshold for acceptance of pixels
 * @param gaussianAdaptive Adaptive thresholding mode (Gaussian or mean)
 * @param binaryMode binary thresholding value
 * @param neighborhoodSize Size of a pixel neighborhood that is used to calculate a threshold value for the pixel
 * @param subtractConstant Constant subtracted from the mean or weighted mean
 * @param newImage
 * @return
 */
ImageObject ImageObject::thresholdAdaptive(int assignmentValue, bool gaussianAdaptive, const std::string &binaryMode,
                                           int neighborhoodSize, int subtractConstant, bool newImage) {

    //
    cv::Mat output;

    // Remap string to int
    std::map<std::string, int> retrieval;
    retrieval["binary"] = 0;
    retrieval["binary_inv"] = 1;
    retrieval["trunc"] = 2;
    retrieval["to_zero"] = 3;
    retrieval["to_zero_inv"] = 4;

    // Set the adaptiveMode
    int adaptiveMode;
    if (gaussianAdaptive){
        adaptiveMode = 1;
    } else{
        adaptiveMode = 0;
    }

    // Binary threshold the image
    if (retrieval.count(binaryMode) == 0){
        std::cout << "We failed to find a given retrieval mode for " << binaryMode << std::endl;
    } else {
        int mode = retrieval[binaryMode];
        cv::adaptiveThreshold(image, output, assignmentValue, adaptiveMode, mode, neighborhoodSize,
                subtractConstant);
    }
    return updateOrExport(output, newImage);
}

/**
 * Isolate elements based on a bgr range
 * @param lowerThresh lower bgra range
 * @param upperThresh upper bgra range
 * @param newImage new image or not
 * @return
 */
ImageObject ImageObject::maskOnColourRange(const cv::Scalar& lowerThresh, const cv::Scalar& upperThresh,
        bool newImage) {
    cv::Mat output;
    cv::inRange(image, lowerThresh, upperThresh, output);
    return updateOrExport(output, newImage);
}







