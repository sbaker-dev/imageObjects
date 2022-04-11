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

    void showImage(const std::string & windowName="Window", int delay=0);

};

#endif //IMAGEOBJECTS_IMAGEOBJECT_H
