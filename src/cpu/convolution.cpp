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

std::pair<double, double> conv_mask(cv::Mat image, int x, int y, int conv_size, int mask1[][3], int mask2[][3])
{
    double sum1 = 0.0;
    double sum2 = 0.0;
    int u = 0;
    int v = 0;
    double cnt1 = 0;
    double cnt2 = 0;
    for (int j = y - conv_size; j <= y + conv_size; j++)
    {
        for (int i = x - conv_size; i <= x + conv_size; i++)
        {
            if (i >= 0 and j >= 0)
            {
                int weight1 = mask1[u][v];
                int weight2 = mask2[u][v];
                auto pix = image.at<uchar>(i, j);
                sum1 += pix * weight1;
                sum2 += pix * weight2;
                cnt1 += abs(weight1);
                cnt2 += abs(weight2);
            }
            v++;
        }
        u++;
        v = 0;
    }
    //sum1 /= cnt1;
    //sum2 /= cnt2;
    double res = sqrt(pow(sum1, 2) + pow(sum2, 2));
    double d = atan2(sum2, sum1);
    d = (d > 0 ? d : (2 * M_PI + d)) * 360 / (2 * M_PI);
    return std::pair<double, double>(res, d);
}

cv::Mat conv_with_mask(cv::Mat image, int conv_size)
{
    auto grad = image.clone();
    auto direction = image.clone();
    int mask1[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    int mask2[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    for (int y = 0; y < image.cols; y++)
    {
        for (int x = 0; x < image.rows; x++)
        {
           auto pair = conv_mask(image, x, y, conv_size, mask1, mask2);
           grad.at<uchar>(x, y) = pair.first;
           std::cout << pair.second << std::endl;
           direction.at<uchar>(x, y) = pair.second;
        }
    }
    return direction;
}

