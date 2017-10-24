#include <iostream>
#include <math.h>
#include <valarray>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

std::valarray<float> conv(Mat image, int y, int x, int conv_size)
{
    std::valarray<float> rgb = {0, 0, 0};
    int cnt = 0;
    for (int j = y - conv_size; j < y + conv_size && j < image.cols; j++)
        for (int i = x - conv_size; i < x + conv_size && i < image.rows; i++)
        {
            if (i >= 0 and j >= 0)
            {
                cnt++;
                rgb[0] += image.at<cv::Vec3b>(j, i)[0];
                rgb[1] += image.at<cv::Vec3b>(j, i)[1];
                rgb[2] += image.at<cv::Vec3b>(j, i)[2];
            }
        }
    if (cnt > 0) {
        rgb /= cnt;
    }
    return rgb;
}

std::valarray<float> gauss_weight(Mat image, int y1, int x1, int y2, int x2, int conv_size, double stddev, float weight_decay)
{
    auto b1 = conv(image, y1, x1, conv_size);
    auto b2 = conv(image, y2, x2, conv_size);
    std::valarray<float> res = {0, 0, 0};
    res = std::exp(-(std::pow(std::abs(b1 - b2), 2) / (std::pow(stddev, 2) * weight_decay)));
    return res;
}

 std::valarray<float> cp(Mat image, int y, int x, int conv_size, double stddev, float weight_decay)
{
    std::valarray<float> sum = {0, 0, 0};
    for (int j = 0; j < image.cols; j++)
        for (int i = 0; i < image.rows; i++)
        {
            auto tmp = gauss_weight(image, y, x, j, i, conv_size, stddev, weight_decay);
            sum += tmp;
        }
    return sum;
}

std::valarray<float> gauss_product(Mat image, Mat nlm_img, int y, int x, int conv_size, double stddev, float weight_decay)
{
    std::valarray<float> sum = {0, 0, 0};

    for (int j = 0; j < nlm_img.cols; j++)
        for (int i = 0; i < nlm_img.rows; i++)
        {
            // f(p, q)
            auto tmp = gauss_weight(nlm_img, y, x, j, i, conv_size, stddev, weight_decay);
            tmp[0] *= (float)image.at<cv::Vec3b>(j, i)[0];
            tmp[1] *= (float)image.at<cv::Vec3b>(j, i)[1];
            tmp[2] *= (float)image.at<cv::Vec3b>(j, i)[2];

            //sum(f(q(
            sum += tmp;
        }
    return sum;
}

Mat non_local_means_cpu(Mat image, int conv_size, float weight_decay)
{
    auto nlm_img = image.clone();
    Scalar mean, stddev;
    cv::meanStdDev(image, mean, stddev);
    double stddev_val = stddev.val[0];
    for (int j = 0; j < nlm_img.cols; j++)
        for (int i = 0; i < nlm_img.rows; i++)
        {
            //C(p)
            auto cp_var = cp(image, j, i, conv_size, stddev_val, weight_decay);

            //sum(v(q) * f(p,q)
            auto tmp = gauss_product(image, nlm_img, j, i, conv_size, stddev_val, weight_decay);

            //sum(v(q) * f(p,q) / C(p)
            tmp /= cp_var;
            nlm_img.at<cv::Vec3b>(j, i)[0] = (int)tmp[0];
            nlm_img.at<cv::Vec3b>(j, i)[1] = (int)tmp[1];
            nlm_img.at<cv::Vec3b>(j, i)[2] = (int)tmp[2];
        }
    return nlm_img;
}

