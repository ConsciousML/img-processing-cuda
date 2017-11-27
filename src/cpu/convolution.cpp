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

cv::Mat non_max_suppr(cv::Mat image, double *grad, double *dir, int threshold)
{
    for (int y = 0; y < image.cols; y++)
    {
        for (int x = 0; x < image.rows; x++)
        {
            double curr_dir = dir[x + y * image.rows];
            double curr_grad = grad[x + y * image.rows];
            if (22.5 <= curr_dir and curr_dir < 67.5
                    and (x - 1) >= 0
                    and (y + 1) < image.cols
                    and (x + 1) < image.rows
                    and (y - 1) >= 0)
            {
                double val1 = grad[(x - 1) + (y + 1) * image.rows];
                double val2 = grad[(x + 1) + (y - 1) * image.rows];
                if (val1 < curr_grad and val2 < curr_grad and curr_grad > 150)
                    image.at<uchar>(x, y) = 255;
                else
                    image.at<uchar>(x, y) = 0;
            }
            else if (67.5 <= curr_dir and curr_dir < 112.5
                    and (x - 1) >= 0
                    and (x + 1) < image.rows)
            {
                double val1 = grad[(x - 1) + y * image.rows];
                double val2 = grad[(x + 1) + y * image.rows];
                if (val1 < curr_grad and val2 < curr_grad and curr_grad > 150)
                    image.at<uchar>(x, y) = 255;
                else
                    image.at<uchar>(x, y) = 0;
            }
            else if (112.5 <= curr_dir and curr_dir < 157.5
                    and (x - 1) >= 0
                    and (y - 1) >= 0
                    and (x + 1) < image.rows
                    and (y + 1) < image.cols)
            {
                double val1 = grad[(x - 1) + (y - 1) * image.rows];
                double val2 = grad[(x + 1) + (y + 1) * image.rows];
                if (val1 < curr_grad and val2 < curr_grad and curr_grad > 150)
                    image.at<uchar>(x, y) = 255;
                else
                    image.at<uchar>(x, y) = 0;
            }
            else if (((0 <= curr_dir and curr_dir < 22.5)
                    or (157.5 <= curr_dir and curr_dir <= 180.0))
                    and (y - 1) >= 0
                    and (y + 1) < image.cols)
            {
                double val1 = grad[x + (y - 1) * image.rows];
                double val2 = grad[x + (y + 1) * image.rows];
                if (val1 < curr_grad and val2 < curr_grad and curr_grad > 150)
                    image.at<uchar>(x, y) = 255;
                else
                    image.at<uchar>(x, y) = 0;
            }
        }
    }
    return image;
}

cv::Mat conv_with_mask(cv::Mat image, int conv_size)
{

    auto edge_image = image.clone();
    double *grad = new double[image.cols * image.rows];
    double *dir = new double[image.cols * image.rows];
    int mask1[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    int mask2[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    for (int y = 0; y < image.cols; y++)
    {
        for (int x = 0; x < image.rows; x++)
        {
           auto pair = conv_mask(image, x, y, conv_size, mask1, mask2);
           grad[x + y * image.rows] = pair.first;
           dir[x + y * image.rows] = pair.second / 2;
        }
    }
    edge_image = non_max_suppr(edge_image, grad, dir);
    return edge_image;
}

