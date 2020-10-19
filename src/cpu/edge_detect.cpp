#include <math.h>
#include "edge_detect.hh"

// Applies a convolution at a given location x, y.
// where conv_size is the size of the convolution mask
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

// Applies a convolution on an entire image
// where conv_size is the size of the convolution mask
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

// Applies a convolution given an image, a location and a mask
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
    double res = sqrt(pow(sum1, 2) + pow(sum2, 2));
    double d = atan2(sum2, sum1);
    d = (d > 0 ? d : (2 * M_PI + d)) * 360 / (2 * M_PI);
    return std::pair<double, double>(res, d);
}

// Classifies the gradient direction for each edges
cv::Mat non_max_suppr(cv::Mat& image, double *grad, double *dir, double thresh)
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
                if (val1 < curr_grad and val2 < curr_grad and curr_grad > thresh)
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
                if (val1 < curr_grad and val2 < curr_grad and curr_grad > thresh)
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
                if (val1 < curr_grad and val2 < curr_grad and curr_grad > thresh)
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
                if (val1 < curr_grad and val2 < curr_grad and curr_grad > thresh)
                    image.at<uchar>(x, y) = 255;
                else
                    image.at<uchar>(x, y) = 0;
            }
        }
    }
    return image;
}

// Computes the gradient of edges and fixes broken edges
bool hysterysis(cv::Mat& image, double *grad, double *dir, double t)
{
    bool changed = false;
    for (int y = 0; y < image.cols; y++)
    {
        for (int x = 0; x < image.rows; x++)
        {
            if (image.at<uchar>(x, y) == 255)
                continue;
            double curr_dir = dir[x + y * image.rows];
            double curr_grad = grad[x + y * image.rows];
            if (22.5 <= curr_dir and curr_dir < 67.5
                    and (x - 1) >= 0
                    and (y + 1) < image.cols
                    and (x + 1) < image.rows
                    and (y - 1) >= 0)
            {
                double dir1 = dir[(x - 1) + (y + 1) * image.rows];
                double dir2 = dir[(x + 1) + (y - 1) * image.rows];
                if (((22.5 <= dir1 and dir1 < 67.5) or (22.5 <= dir2 and dir2 < 67.5)) and curr_grad > t)
                {
                    image.at<uchar>(x, y) = 255;
                    changed = true;
                }
            }
            else if (67.5 <= curr_dir and curr_dir < 112.5
                    and (x - 1) >= 0
                    and (x + 1) < image.rows)
            {
                double dir1 = grad[(x - 1) + y * image.rows];
                double dir2 = grad[(x + 1) + y * image.rows];
                if (((67.5 <= dir1 and dir1 < 112.5) or (67.5 < dir2 and dir2 < 112.5)) and curr_grad > t)
                {
                    image.at<uchar>(x, y) = 255;
                    changed = true;
                }
            }
            else if (112.5 <= curr_dir and curr_dir < 157.5
                    and (x - 1) >= 0
                    and (y - 1) >= 0
                    and (x + 1) < image.rows
                    and (y + 1) < image.cols)
            {
                double dir1 = grad[(x - 1) + (y - 1) * image.rows];
                double dir2 = grad[(x + 1) + (y + 1) * image.rows];
                if (((112.5 <= dir1 and dir1 < 157.5) or (112.5 <= dir2 and dir2 < 157.5)) and curr_grad > t)
                {
                    image.at<uchar>(x, y) = 255;
                    changed = true;
                }
            }
            else if (((0 <= curr_dir and curr_dir < 22.5)
                    or (157.5 <= curr_dir and curr_dir <= 180.0))
                    and (y - 1) >= 0
                    and (y + 1) < image.cols)
            {
                double dir1 = grad[x + (y - 1) * image.rows];
                double dir2 = grad[x + (y + 1) * image.rows];
                if ((((0 <= dir1 and dir1 < 22.5) and (157.5 <= dir1 and dir1 <= 180.5))
                    or ((0 <= dir2 and dir2 < 22.5) and (157.5 <= dir2 and dir2 <= 180.5))) and curr_grad > t)
                {
                    image.at<uchar>(x, y) = 255;
                    changed = true;
                }
            }
        }
    }
    return changed;
}

// Applies a convolution on an entire image given a convolution mask
cv::Mat conv_with_mask(cv::Mat image, int conv_size)
{
    cv::Mat dst;
    cv::Mat tmp_image;
    double otsu_threshold = cv::threshold(image, dst, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
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
    non_max_suppr(edge_image, grad, dir, otsu_threshold);
    bool changed = hysterysis(edge_image, grad, dir, otsu_threshold * 0.5);
    std::cout << changed << std::endl;
    while (changed)
    {
        changed = hysterysis(edge_image, grad, dir, otsu_threshold * 0.5);
        std::cout << changed << std::endl;
    }
    return edge_image;
}

