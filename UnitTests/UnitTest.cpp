#include "../src/ImageObject.h"

#include "opencv2/core/utils/logger.hpp"
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <iostream>
#include <utility>


class UnitTest{

public:
    std::string loadPath;

    explicit UnitTest(std::string path){
        loadPath = std::move(path);
    }

    int runTests(){

        // Initialise the ImageObject
        ImageObject img = ImageObject(cv::imread(loadPath));

        // Check the dimensions of the image, therefore also checking it has loaded
        testDimensions(img);

        // Test Inversion
        testInvert(img);

        // Test Contours
        testContours(img);

        // Test Thresh-holding
        testThreshold(img);




        return 0;
    }

private:
    /**
     * Test that the image has not changed size by validating its known dimensions
     * @param img Test Image
     */
    static void testDimensions(ImageObject img){

        // Test Status
        bool failedTest = false;

        // Validate the image is not empty
        if (!img.empty()){
            std::cout << "Expect to find an image, but the image was empty" <<std::endl;
            failedTest = true;
        }

        // Check how many pixels are not equal to zero
        if (img.nonZero() != 218119){
            std::cout << "Expected to find 218119 non zero pixels yet found" + std::to_string(img.nonZero())
                << std::endl;
            failedTest = true;
        }

        // Check Width
        if (img.width() != 574){
            std::cout << "Expected width of 574 yet found " + std::to_string(img.width()) << std::endl;
            failedTest = true;
        }

        // Check Height
        if (img.height() != 380){
            std::cout << "Expected width of 380 yet found " + std::to_string(img.height()) << std::endl;
            failedTest = true;
        }

        // Validate the these two numbers do correspond to the defined pixel total
        if (img.pixelTotal() != 218120){
            std::cout << "Expected total pixel count of 218120 yet found " + std::to_string(img.pixelTotal())
                << std::endl;
            failedTest = true;
        }

        if (img.channels() != 3){
            std::cout << "Expect to find 3 channels yet found " + std::to_string(img.channels()) << std::endl;
            failedTest = true;
        }

        // Validate result of Test
        if (failedTest){
            std::cout << "\tFailed: Dimensions" << std::endl;
        } else {
            std::cout << "Success: Dimensions" << std::endl;
        }
    }

    /**
     * Invert the image and test the first row matches the known value totals of BGRA
     * @param img
     */
    static void testInvert(ImageObject img){

        // Invert the image as a new instance
        ImageObject inverted = img.invert(true);

        // Extract the first row and validate that the sum of the rows resulting BGRA Vector 4 is equal to the known
        // value
        cv::Scalar_<double> invertTotal = cv::Scalar_<double>(17379, 23935, 26712, 0);
        cv::Scalar_<double> invertValidate = cv::sum(inverted.extractRow(0));

        // Check status
        if (invertValidate != invertTotal){
            std::cout << "\tFailed: Inversion" << std::endl;
        }
        else{
            std::cout << "Success: Inversion" << std::endl;
        }
    }

    /**
     * Test contour functionality
     * @param img
     */
    static void testContours(ImageObject img){

        // Test Status
        bool failedTest = false;

        // Extract the contours
        std::vector<std::vector<cv::Point>> contours = img.extractContours("external");

        // Check the values of each contour in the vector is equal to a known value
        for(const auto& value: contours) {
            if (value[0] != cv::Point(0, 0)){
                std::cout << "First point failed" << std::endl;
                failedTest = true;
            }
            if (value[1] != cv::Point(0, 379)){
                std::cout << "Second point failed" << std::endl;
                failedTest = true;
            }
            if (value[2] != cv::Point(573, 379)){
                std::cout << "Third point failed" << std::endl;
                failedTest = true;
            }
            if (value[3] != cv::Point(573, 0)){
                std::cout << "Forth point failed" << std::endl;
                failedTest = true;
            }
        }

        // Test Draw contour
        img.drawContour(contours, -1, cv::Scalar(0, 255, 0), 2, true);

        // Validate result of Test
        if (failedTest){
            std::cout << "\tFailed: Contours" << std::endl;
        } else {
            std::cout << "Success: Contours" << std::endl;
        }

    }

    static void testThreshold(ImageObject img){

        // Create an image for binary checking
        ImageObject binary = img.changeToMono(true);
        binary.thresholdBinary(100);

        // Extract the first row and validate that the sum of the rows resulting BGRA Vector 4 is equal to the known
        // value
        cv::Scalar_<double> thresholdTotal = cv::Scalar_<double>(146370, 0, 0, 0);
        cv::Scalar_<double> thresholdValidate = cv::sum(binary.extractRow(0));

        // Check status
        if (thresholdValidate != thresholdTotal){
            std::cout << "\tFailed: Binary Inversion" << std::endl;
        }
        else{
            std::cout << "Success: Binary Inversion" << std::endl;
        }


        // Create an image for adaptive threshold checking
        ImageObject adaptive = img.changeToMono(true);
        adaptive.thresholdAdaptive();

        // Extract the first col and validate that the sum of the rows resulting BGRA Vector 4 is equal to the known
        // value
        cv::Scalar_<double> adaptiveThresholdTotal = cv::Scalar_<double>(87210, 0, 0, 0);
        cv::Scalar_<double> adaptiveThresholdValidate = cv::sum(adaptive.extractCol(0));

        // Check status
        if (adaptiveThresholdValidate != adaptiveThresholdTotal){
            std::cout << "\tFailed: Adaptive Inversion" << std::endl;
        }
        else{
            std::cout << "Success: Adaptive Inversion" << std::endl;
        }

    }

};



int main(){

    // Silence the CV log
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);

    // Start the testing
    std::cout << "Starting Test Frame work" << std::endl;

    // Redefine relative to your build directory
    std::string testImagePath = R"(C:\Users\Samuel\CLionProjects\imageObjects\UnitTests\WastWater.jpg)";

    // Initialise the testing framework
    UnitTest testFrameWork = UnitTest(testImagePath);

    // Run the tests
    testFrameWork.runTests();

    return 0;
}
