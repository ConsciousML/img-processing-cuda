#include <valarray>
#include <iostream>
#include <stdio.h>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "kernel.hh"


int main(int argc, char** argv)
{
    if (argc < 3)
    {
        printf("usage: main <Image_Path> <Conv_size>\n");
        return 1;
    }
    cv::Mat image;
    image = cv::imread(argv[1], CV_LOAD_IMAGE_UNCHANGED);
    if (!image.data)
    {
        std::cout << "Could not open or find the image" << std::endl;
        return 1;
    }
    /*int conv_size = stoi(argv[2]);
    auto conv_image = image.clone();
    for (int j = 0; j < image.cols; j++)
        for (int i = 0; i < image.rows; i++)
        {
            auto res = conv(image, j, i, conv_size);
            conv_image.at<cv::Vec3b>(j, i)[0] = res[0];
            conv_image.at<cv::Vec3b>(j, i)[1] = res[1];
            conv_image.at<cv::Vec3b>(j, i)[2] = res[2];
        }
    */
    //conv(image);
    //cv::namedWindow("Display Window", CV_WINDOW_AUTOSIZE);
    //cv::imshow("Display Window", image);
    cv::waitKey(0);
    test_kernel();
    return 0;
}
