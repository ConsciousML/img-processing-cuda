#include <iostream>
#include <math.h>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

std::tuple<float, float, float> conv(Mat image, int x, int y, int conv_size)
{
    std::tuple<float, float, float> rgb;
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
    if (cnt > 0) {
        std::get<0>(rgb) /= cnt;
        std::get<1>(rgb) /= cnt;
        std::get<2>(rgb) /= cnt;
    }
    return rgb;
}

std::tuple<float, float, float> gauss_weight(Mat image, int x1, int y1, int x2, int y2, int conv_size)
{
    auto b1 = conv(image, x1, y1, conv_size);
    auto b2 = conv(image, x2, y2, conv_size);
    std::get<0>(b1) -= std::get<0>(b2);
    std::get<1>(b1) -= std::get<1>(b2);
    std::get<2>(b1) -= std::get<2>(b2);

    std::get<0>(b1) = std::abs(std::get<0>(b1));
    std::get<1>(b1) = std::abs(std::get<1>(b1));
    std::get<2>(b1) = std::abs(std::get<2>(b1));

    std::get<0>(b1) = std::pow(std::get<0>(b1), 2);
    std::get<1>(b1) = std::pow(std::get<1>(b1), 2);
    std::get<2>(b1) = std::pow(std::get<2>(b1), 2);

    std::get<0>(b1) /= std::pow(10, 2);
    std::get<1>(b1) /= std::pow(10, 2);
    std::get<2>(b1) /= std::pow(10, 2);

    std::get<0>(b1) = std::exp(std::get<0>(b1) * (-1));
    std::get<1>(b1) = std::exp(std::get<1>(b1) * (-1));
    std::get<2>(b1) = std::exp(std::get<2>(b1) * (-1));
    return b1;
}

 std::tuple<float, float, float> cp(Mat image, int x, int y, int conv_size)
{
    std::tuple<float, float, float> sum;
    std::get<0>(sum) = 0;
    std::get<1>(sum) = 0;
    std::get<2>(sum) = 0;
    for (int j = 0; j < image.cols; j++)
        for (int i = 0; i < image.rows; i++)
        {
            auto tmp = gauss_weight(image, x, y, i, j, conv_size);
            std::get<0>(sum) += std::get<0>(tmp);
            std::get<1>(sum) += std::get<1>(tmp);
            std::get<2>(sum) += std::get<2>(tmp);
        }
    return sum;
}

std::tuple<float, float, float> gauss_product(Mat image, Mat nlm_img, int x, int y, int conv_size)
{
    std::tuple<float, float, float> sum;
    std::get<0>(sum) = 0;
    std::get<1>(sum) = 0;
    std::get<2>(sum) = 0;

    for (int j = 0; j < nlm_img.cols; j++)
        for (int i = 0; i < nlm_img.rows; i++)
        {
            // f(p, q)
            auto tmp = gauss_weight(nlm_img, x, y, i, j, conv_size);
            std::get<0>(tmp) *= image.at<cv::Vec3b>(i, j)[0];
            std::get<1>(tmp) *= image.at<cv::Vec3b>(i, j)[1];
            std::get<2>(tmp) *= image.at<cv::Vec3b>(i, j)[2];

            //sum(f(q(
            std::get<0>(sum) += std::get<0>(tmp);
            std::get<1>(sum) += std::get<1>(tmp);
            std::get<2>(sum) += std::get<2>(tmp);
        }
    return sum;
}

Mat non_local_means_cpu(Mat image, int conv_size)
{
    auto nlm_img = image.clone();
    for (int j = 0; j < nlm_img.cols; j++)
        for (int i = 0; i < nlm_img.rows; i++)
        {
            // C(p)
            auto cp_var = cp(image, i, j, conv_size);

            //sum(v(q) * f(p,q)
            auto tmp = gauss_product(image, nlm_img, i, j, conv_size);

            //sum(v(q) * f(p,q) / C(p)
            std::get<0>(tmp) /= std::get<0>(cp_var);
            std::get<1>(tmp) /= std::get<1>(cp_var);
            std::get<2>(tmp) /= std::get<2>(cp_var);

            nlm_img.at<cv::Vec3b>(i, j)[0] = std::get<0>(tmp);
            nlm_img.at<cv::Vec3b>(i, j)[1] = std::get<1>(tmp);
            nlm_img.at<cv::Vec3b>(i, j)[2] = std::get<2>(tmp);
        }
    return nlm_img;
}

