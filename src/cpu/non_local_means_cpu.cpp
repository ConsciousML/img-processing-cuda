#include <iostream>
#include <math.h>
#include <valarray>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

// Ecart-type
/*float get_deviation(Mat image)
{
    float mean = 0;
    for (int j = 0; j < nlm_img.cols; j++)
        for (int i = 0; i < nlm_img.rows; i++)
    {
        
    }
}*/

std::valarray<float> conv(Mat image, int x, int y, int conv_size)
{
    std::valarray<float> rgb = {0, 0, 0};
    int cnt = 0;
    for (int j = x - conv_size; j < x + conv_size && j < image.cols; j++)
        for (int i = y - conv_size; i < y + conv_size && i < image.rows; i++)
        {
            if (i >= 0 and j >= 0)
            {
                cnt++;
                rgb[0] += image.at<cv::Vec3b>(i, j)[0];
                rgb[1] += image.at<cv::Vec3b>(i, j)[1];
                rgb[2] += image.at<cv::Vec3b>(i, j)[2];
            }
        }
    if (cnt > 0) {
        rgb /= cnt;
    }
    return rgb;
}

std::valarray<float> gauss_weight(Mat image, int x1, int y1, int x2, int y2, int conv_size)
{
    auto b1 = conv(image, x1, y1, conv_size);
    auto b2 = conv(image, x2, y2, conv_size);
    std::valarray<float> res = {0, 0, 0};
    res = std::exp(-(std::pow(std::abs(b1 - b2), 2) / std::pow(10, 2)));
    return res;
}

 std::valarray<float> cp(Mat image, int x, int y, int conv_size)
{
    std::valarray<float> sum = {0, 0, 0};
    for (int j = 0; j < image.cols; j++)
        for (int i = 0; i < image.rows; i++)
        {
            auto tmp = gauss_weight(image, x, y, i, j, conv_size);
            sum += tmp;
        }
    return sum;
}

std::valarray<float> gauss_product(Mat image, Mat nlm_img, int x, int y, int conv_size)
{
    std::valarray<float> sum = {0, 0, 0};

    for (int j = 0; j < nlm_img.cols; j++)
        for (int i = 0; i < nlm_img.rows; i++)
        {
            // f(p, q)
            auto tmp = gauss_weight(nlm_img, x, y, i, j, conv_size);
            tmp[0] *= (float)image.at<cv::Vec3b>(i, j)[0];
            tmp[1] *= (float)image.at<cv::Vec3b>(i, j)[1];
            tmp[2] *= (float)image.at<cv::Vec3b>(i, j)[2];

            //sum(f(q(
            sum += tmp;
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
            tmp /= cp_var;
            nlm_img.at<cv::Vec3b>(i, j)[0] = (int)tmp[0];
            nlm_img.at<cv::Vec3b>(i, j)[1] = (int)tmp[1];
            nlm_img.at<cv::Vec3b>(i, j)[2] = (int)tmp[2];
        }
    return nlm_img;
}

