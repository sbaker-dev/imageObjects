//
// Created by Samuel on 11/04/2022.
//

#include "../src/ImageObject.h"
#include <opencv2/opencv.hpp>
#include <iostream>


int main(){
    std::cout << "Hello, World!" << std::endl;

    ImageObject img = ImageObject(cv::imread("C:/Users/Samuel/Pictures/stopper.jpg"));

    img.showImage();

    return 0;
};