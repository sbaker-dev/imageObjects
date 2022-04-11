#ifndef IMAGEOBJECTS_IMAGEOBJECT_H
#define IMAGEOBJECTS_IMAGEOBJECT_H

#include <opencv2/opencv.hpp>


class ImageObject{
public:
    cv::Mat image;

    explicit ImageObject(cv::Mat Image);

    int height();

    int width();

    int pixelTotal();

    cv::Mat extractRow(int rowIndex);

    cv::Mat extractCol(int colIndex);

    ImageObject updateOrExport(const cv::Mat& img, bool newImage);

    ImageObject invert(bool newImage=false);

    void showImage(const std::string & windowName="Window", int delay=0);

};

#endif //IMAGEOBJECTS_IMAGEOBJECT_H
