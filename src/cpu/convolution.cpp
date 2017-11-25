#include <math.h>
#include "convolution.hh"
std::valarray<double> simple_conv(cv::Mat image, int x, int y, int conv_size)
{
    std::valarray<double> rgb = {0, 0, 0};
    int cnt = 0;
    for (int j = y - conv_size; j < y + conv_size; j++)
    {
        for (int i = x - conv_size; i < x + conv_size; i++)
        {
            if (i >= 0 and j >= 0)
            {
		cnt++;
                rgb[0] += image.at<cv::Vec3b>(i, j)[0];
                rgb[1] += image.at<cv::Vec3b>(i, j)[1];
                rgb[2] += image.at<cv::Vec3b>(i, j)[2];
            }
        }
    }
    rgb[0] /= cnt;
    rgb[1] /= cnt;
    rgb[2] /= cnt;
    return rgb;
}

cv::Mat convolution(cv::Mat image, int conv_size)
{
    auto conv_img = image.clone();
    for (int y = 0; y < image.cols; y++)
    {
        for (int x = 0; x < image.rows; x++)
        {
           auto gc = simple_conv(image, x, y, conv_size);
           conv_img.at<cv::Vec3b>(x, y)[0] = gc[0];
           conv_img.at<cv::Vec3b>(x, y)[1] = gc[1];
           conv_img.at<cv::Vec3b>(x, y)[2] = gc[2];
        }
    }
    return conv_img;
}

std::valarray<double> conv_mask(cv::Mat image, int x, int y, int conv_size, int mask1[][3], int mask2[][3])
{
    std::valarray<double> rgb1 = {0, 0, 0};
    std::valarray<double> rgb2 = {0, 0, 0};
    int u = 0;
    int v = 0;
    for (int j = y - conv_size; j <= y + conv_size; j++)
    {
        for (int i = x - conv_size; i <= x + conv_size; i++)
        {
            if (i >= 0 and j >= 0)
            {
                int weight1 = mask1[u][v];
                int weight2 = mask2[u][v];
                auto pix = image.at<cv::Vec3b>(i, j);
                rgb1[0] += pix[0] * weight1;
                rgb1[1] += pix[1] * weight1;
                rgb1[2] += pix[2] * weight1;
                rgb2[0] += pix[0] * weight2;
                rgb2[1] += pix[1] * weight2;
                rgb2[2] += pix[2] * weight2;
            }
            v++;
        }
        u++;
        v = 0;
    }
    std::valarray<double> res = {sqrt(pow(rgb1[0], 2) + pow(rgb2[0],2)),
                                 sqrt(pow(rgb1[1], 2) + pow(rgb2[1],2)),
                                 sqrt(pow(rgb1[2], 2) + pow(rgb2[2],2))};
    return res;
}

cv::Mat conv_with_mask(cv::Mat image, int conv_size)
{
    auto img = image.clone();
    int mask1[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    int mask2[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    for (int y = 0; y < image.cols; y++)
    {
        for (int x = 0; x < image.rows; x++)
        {
           auto gc = conv_mask(image, x, y, conv_size, mask1, mask2);
           img.at<cv::Vec3b>(x, y)[0] = gc[0];
           img.at<cv::Vec3b>(x, y)[1] = gc[1];
           img.at<cv::Vec3b>(x, y)[2] = gc[2];
        }
    }
    return img;
}

