#include <math.h>
#include "convolution.hh"
std::valarray<double> conv_mask(cv::Mat image, int x, int y, int conv_size)
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
           auto gc = conv_mask(image, x, y, conv_size);
           conv_img.at<cv::Vec3b>(x, y)[0] = gc[0];
           conv_img.at<cv::Vec3b>(x, y)[1] = gc[1];
           conv_img.at<cv::Vec3b>(x, y)[2] = gc[2];
        }
    }
    return conv_img;
}

