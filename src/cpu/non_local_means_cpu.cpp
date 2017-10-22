#include <iostream>
#include <math.h>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

std::tuple<int, int, int> conv(Mat image, int x, int y, int conv_size)
{
    std::tuple<int, int, int> rgb;
    std::get<0>(rgb) = 0;
    std::get<1>(rgb) = 0;
    std::get<2>(rgb) = 0;
    int cnt = 0;
    for (int j = x - conv_size; j < x + conv_size && j < image.cols; j++)
        for (int i = y - conv_size; i < y + conv_size && i < image.rows; i++)
        {
            if (i >= 0 and j >= 0)
            {
                cnt++;
                std::get<0>(rgb) += image.at<cv::Vec3b>(i, j)[0];
                std::get<1>(rgb) += image.at<cv::Vec3b>(i, j)[1];
                std::get<2>(rgb) += image.at<cv::Vec3b>(i, j)[2];
            }
    }
    std::get<0>(rgb) /= cnt;
    std::get<1>(rgb) /= cnt;
    std::get<2>(rgb) /= cnt;
    return rgb;
}

Mat non_local_means_cpu(Mat image, int conv_size)
{
    auto nlm_img = image.clone();
    for (int j = 0; j < nlm_img.cols; j++)
        for (int i = 0; i < nlm_img.rows; i++)
        {
            auto rgb = conv(image, j, i, conv_size);
            nlm_img.at<cv::Vec3b>(i, j)[0] = std::get<0>(rgb);
            nlm_img.at<cv::Vec3b>(i, j)[1] = std::get<1>(rgb);
            nlm_img.at<cv::Vec3b>(i, j)[2] = std::get<2>(rgb);
        }
    return nlm_img;
}

