#ifndef IMAGEOBJECTS_IMAGEOBJECT_H
#define IMAGEOBJECTS_IMAGEOBJECT_H

#include <opencv2/opencv.hpp>


class ImageObject{
public:
    cv::Mat image;

    explicit ImageObject(cv::Mat Image);

    ImageObject updateOrExport(const cv::Mat& img, bool newImage);

    void showImage(const std::string & windowName="Window", int delay=0);

    int height();

    int width();

    int pixelTotal();

    int channels();

    int nonZero();

    bool empty();

    cv::Mat extractRow(int rowIndex);

    cv::Mat extractCol(int colIndex);

    std::vector<std::vector<cv::Point>> extractContours(const std::string& retrieval_mode, bool simple_approx=true);

    ImageObject changeBgrToRgb(bool newImage=false);

    ImageObject changeBgraToBgr(bool newImage=false);

    ImageObject changeBgrToBgra(bool newImage=false);

    ImageObject changeToColour(bool newImage=false);

    ImageObject changeToMono(bool newImage=false);

    ImageObject invert(bool newImage=false);

    ImageObject drawContour(const std::vector<std::vector<cv::Point>>& contours, int index, const cv::Scalar& colour,
            int thickness, bool newImage=false);

    ImageObject thresholdBinary(int binaryThreshold, const std::string& binaryMode = "binary", int binaryMax=255,
            bool newImage=false);

    ImageObject thresholdAdaptive(int assignmentValue=255, bool gaussianAdaptive=true,
            const std::string& binaryMode = "binary", int neighborhoodSize=51, int subtractConstant=20,
            bool newImage=false);


};

#endif //IMAGEOBJECTS_IMAGEOBJECT_H
