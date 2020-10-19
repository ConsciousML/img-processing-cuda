#include <math.h>
#include "knn.hh"

using namespace std;
using namespace cv;

// Applies a convolution given two points
std::valarray<double> conv(Mat image, int x1, int y1, int x2, int y2, int conv_size)
{
    std::valarray<double> rgb = {0, 0, 0};
    int cnt = 0;
    for (int j1 = y1 - conv_size; j1 < y1 + conv_size; j1++)
        for (int i1 = x1 - conv_size; i1 < x1 + conv_size; i1++)
        {
            int i2 = i1 - x1 + x2;
            int j2 = j1 - y1 + y2;
            if (i1 >= 0 and j1 >= 0 and j2 >= 0 and i2 >= 0 and
                i1 < image.cols and j1 < image.rows and
                i2 < image.cols and j2 < image.rows)
            {
                cnt++;
                auto pix1 = image.at<cv::Vec3b>(j1, i1);
                auto pix2 = image.at<cv::Vec3b>(j2, i2);
                rgb[0] += std::pow(std::abs(pix1[0] - pix2[0]), 2);
                rgb[1] += std::pow(std::abs(pix1[1] - pix2[1]), 2);
                rgb[2] += std::pow(std::abs(pix1[2] - pix2[2]), 2);
            }
        }
    if (cnt > 0) {
        rgb /= cnt;
    }
    return rgb;
}

// Applies a convolution given a location and computes the nlm formula
std::valarray<double> gauss_conv_nlm(cv::Mat image, int x, int y, int conv_size, int block_radius, double h_param)
{
    std::valarray<double> rgb = {0, 0, 0};
    std::valarray<double> cnt = {0, 0, 0};
    int cx = 0;
    for (int j = y - conv_size; j < y + conv_size; j++)
    {
        for (int i = x - conv_size; i < x + conv_size; i++)
        {
            if (i < image.rows and j < image.cols and i >= 0 and j >= 0)
            {
                auto u = conv(image, y, x, j, i, block_radius);
                auto uy = image.at<cv::Vec3b>(i, j);
                double c1 = std::exp(-(std::pow(std::abs(i + j - (x + y)), 2)) / (float)std::pow(conv_size, 2));
                double h_div = std::pow(h_param, 2);

                std::valarray<double> c2 = {std::exp(-u[0] / h_div),
                    std::exp(-u[1] / h_div),
                    std::exp(-u[2] / h_div)};

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

// Hat function for the Non Local-Means algorithm
cv::Mat non_local_means_cpu(cv::Mat image, int conv_size, int block_radius, double h_param)
{
    auto nlm_img = image.clone();
    for (int y = 0; y < image.cols; y++)
    {
        for (int x = 0; x < image.rows; x++)
        {
           auto gc = gauss_conv_nlm(image, x, y, conv_size, block_radius, h_param);
           nlm_img.at<cv::Vec3b>(x, y)[0] = gc[0];
           nlm_img.at<cv::Vec3b>(x, y)[1] = gc[1];
           nlm_img.at<cv::Vec3b>(x, y)[2] = gc[2];
        }
    }
    return nlm_img;
}
