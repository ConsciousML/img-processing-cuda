#include <math.h>
#include "knn.hh"


// Applies a gaussian convolution
std::valarray<double> gauss_conv(cv::Mat image, int x, int y, int conv_size, double h_param)
{
    std::valarray<double> rgb = {0, 0, 0};
    std::valarray<double> cnt = {0, 0, 0};
    int cx = 0;
    for (int j = y - conv_size; j < y + conv_size; j++)
    {
        for (int i = x - conv_size; i < x + conv_size; i++)
        {
            if (i >= 0 and j >= 0)
            {
                auto ux = image.at<cv::Vec3b>(x, y);
                auto uy = image.at<cv::Vec3b>(i, j);
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


// Hat function of the K-Nearest Neighbors algorithm
// Removes the noise of an image
cv::Mat knn(cv::Mat image, int conv_size, double h_param)
{
    auto knn_img = image.clone();
    for (int y = 0; y < image.cols; y++)
    {
        for (int x = 0; x < image.rows; x++)
        {
           auto gc = gauss_conv(image, x, y, conv_size, h_param);
           //auto gc = image.at<cv::Vec3b>(y, x);
           knn_img.at<cv::Vec3b>(x, y)[0] = gc[0];
           knn_img.at<cv::Vec3b>(x, y)[1] = gc[1];
           knn_img.at<cv::Vec3b>(x, y)[2] = gc[2];
        }
    }
    return knn_img;
}


// Applies a gauss convolution given a location and a grey image
double gauss_conv_gray(cv::Mat image, int x, int y, int conv_size, double h_param)
{
    double rgb = 0;
    double cnt = 0;
    int cx = 0;
    for (int j = y - conv_size; j < y + conv_size; j++)
    {
        for (int i = x - conv_size; i < x + conv_size; i++)
        {
            if (i >= 0 and j >= 0)
            {
                auto ux = image.at<uchar>(x, y);
                auto uy = image.at<uchar>(i, j);
                double c1 = std::exp(-(std::pow(std::abs(i + j - (x + y)), 2)) / (float)std::pow(conv_size, 2));
                double h_div = std::pow(h_param, 2);

                double c2 = std::exp(-(std::pow(std::abs(uy - ux), 2)) / h_div);

                rgb += uy * c1 * c2;

                cnt += c1 * c2;
            }
        }
    }
    if (cnt != 0)
    {
        rgb /= cnt;
    }
    return rgb;
}


// Hat function of the K-Nearest Neighbors algorithm on a grey image
// Removes the noise of a grey image
cv::Mat knn_grey(cv::Mat image, int conv_size, double h_param)
{
    auto knn_img = image.clone();
    for (int y = 0; y < image.cols; y++)
    {
        for (int x = 0; x < image.rows; x++)
        {
           auto gc = gauss_conv_gray(image, x, y, conv_size, h_param);
           knn_img.at<uchar>(x, y) = gc;
        }
    }
    return knn_img;
}
