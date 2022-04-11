#include "../src/ImageObject.h"
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

};



int main(){
    std::cout << "Starting Test Frame work" << std::endl;

    // Redefine relative to your build directory
    std::string testImagePath = R"(C:\Users\Samuel\CLionProjects\imageObjects\UnitTests\WastWater.jpg)";

    // Initialise the testing framework
    UnitTest testFrameWork = UnitTest(testImagePath);

    // Run the tests
    testFrameWork.runTests();

    return 0;
}
