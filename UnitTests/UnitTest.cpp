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
        return 0;
    }



private:
    /**
     * Test that the image has not changed size by validating its known dimensions
     * @param img Test Image
     */
    static void testDimensions(ImageObject img){

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

        // Validate Test
        if (failedTest){
            std::cout << "Failed to load originally defined image" << std::endl;
        } else {
            std::cout << "Loaded originally defined image based on dimensions" << std::endl;
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
