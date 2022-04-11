//
// Created by Samuel on 11/04/2022.
//

#include <opencv2/opencv.hpp>
#include <utility>
#include "ImageObject.h"


class ImageObject{
public:
    cv::Mat image;

    explicit ImageObject(cv::Mat Image){
        image = std::move(Image);
    }


    /**
     * Show the current instance of Image
     * @param windowName: The name of the window when shown, defaults to Window
     * @param delay: The delay after which the window closes. If set to zero, which is the default, will wait until
     *      button pressed
     */
    void showImage(const std::string & windowName="Window", int delay=0){
        cv::imshow(windowName, image);
        cv::waitKey(delay);
    }




};



int main() {
    std::cout << "Hello, World!" << std::endl;

    ImageObject imgObj = ImageObject(cv::imread("C:/Users/Samuel/Pictures/stopper.jpg"));


    imgObj.showImage();

//    cv::Mat srcImage = cv::imread("C:/Users/Samuel/Pictures/stopper.jpg");
//
//    cv::imshow("[img]", srcImage);
//    cv::waitKey(0);

    return 0;
}
