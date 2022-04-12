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

    ImageObject changeBgrToRgb(bool newImage=false);

    ImageObject changeBgraToBgr(bool newImage=false);

    ImageObject changeBgrToBgra(bool newImage=false);

    ImageObject changeToColour(bool newImage=false);

    ImageObject changeToMono(bool newImage=false);

    ImageObject invert(bool newImage=false);


};

#endif //IMAGEOBJECTS_IMAGEOBJECT_H
