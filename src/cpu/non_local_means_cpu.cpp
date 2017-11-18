#include <math.h>
#include "knn.hh"

using namespace std;
using namespace cv;

std::valarray<float> conv(Mat image, int x, int y, int conv_size)
{
    std::valarray<float> rgb = {0, 0, 0};
    int cnt = 0;
    for (int j = y - conv_size; j < y + conv_size; j++)
        for (int i = x - conv_size; i < x + conv_size; i++)
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

std::valarray<double> gauss_conv_nlm(cv::Mat image, int x, int y, int conv_size, double h_param)
{
    std::valarray<double> rgb = {0, 0, 0};
    std::valarray<double> cnt = {0, 0, 0};
    int cx = 0;
    auto ux = conv(image, y, x, conv_size);
    for (int j = y - conv_size; j < y + conv_size; j++)
    {
        for (int i = x - conv_size; i < x + conv_size; i++)
        {
            if (i < image.rows and j < image.cols and i >= 0 and j >= 0)
            {
                auto uy = conv(image, j, i, conv_size);
                double c1 = std::exp(-(std::pow(std::abs(i + j - (x + y)), 2)) / (float)std::pow(conv_size, 2));
                double h_div = std::pow(h_param, 2);

                std::valarray<double> c2 = {std::exp(-(std::pow(std::abs(uy[0] - ux[0]), 2)) / h_div),
                    std::exp(-(std::pow(std::abs(uy[1] - ux[1]), 2)) / h_div),
                    std::exp(-(std::pow(std::abs(uy[2] - ux[2]), 2)) / h_div)};

                std::valarray<double> c = {c1, c1, c1};

                rgb[0] += uy[0] * c1 * c2[0];
                rgb[1] += uy[1] * c1 * c2[1];
                rgb[2] += uy[2] * c1 * c2[2];

                cnt[0] += c1 * c2[0];
                cnt[1] += c1 * c2[1];
                cnt[2] += c1 * c2[2];
            }
        }
    }
    if (cnt[0] != 0 and cnt[1] != 0 and cnt[2] != 0)
    {
        rgb[0] /= cnt[0];
        rgb[1] /= cnt[1];
        rgb[2] /= cnt[2];
    }
    return rgb;
}


cv::Mat non_local_means_cpu(cv::Mat image, int conv_size, double h_param)
{
    auto nlm_img = image.clone();
    for (int y = 0; y < image.cols; y++)
    {
        for (int x = 0; x < image.rows; x++)
        {
           auto gc = gauss_conv_nlm(image, x, y, conv_size, h_param);
           nlm_img.at<cv::Vec3b>(x, y)[0] = gc[0];
           nlm_img.at<cv::Vec3b>(x, y)[1] = gc[1];
           nlm_img.at<cv::Vec3b>(x, y)[2] = gc[2];
        }
    }
    return nlm_img;
}
