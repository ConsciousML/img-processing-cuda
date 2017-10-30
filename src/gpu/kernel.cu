#include <stdio.h>
#include <iostream>
#include <valarray>
#include <assert.h>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

struct Rgb {
    Rgb(float x, float y, float z) : r(x), g(y), b(z) {}
    Rgb(cv::Vec3b v) : r(v[0]), g(v[1]), b(v[2]) {}
    void div(int x)
    {
        r /= x;
        g /= x;
        b /= x;
    }
    float r;
    float g;
    float b;
};

__global__ void kernel_conv(Rgb* device_img, Rgb* img, int rows, int cols, int conv_size)
{
    int cnt = 0;
    for (int y = 0; y < cols; y++)
        for (int x = 0; x < rows; x++)
        {
            cnt = 0;
            for (int i = y - conv_size; i < y + conv_size && i < cols; i++)
                for (int j = x - conv_size; j < x + conv_size && j < rows; j++)
                {
                    if (i >= 0 and j >= 0)
                    {
                        cnt++;
                        device_img[x + y * rows].r += img[j + i * rows].r;
                        device_img[x + y * rows].g += img[j + i * rows].g;
                        device_img[x + y * rows].b += img[j + i * rows].b;
                    }
                }
            if (cnt > 0)
            {
                device_img[x + y * rows].r /= cnt;
                device_img[x + y * rows].g /= cnt;
                device_img[x + y * rows].b /= cnt;
            }
        }
}

Rgb *img_to_device(cv::Mat img)
{
    Rgb *device_img;
    int width = img.rows;
    int height = img.cols;
    cudaMallocManaged(&device_img, width * height * sizeof (Rgb));

    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
            device_img[j + i * width] = Rgb(img.at<cv::Vec3b>(j, i));

    return device_img;
}

void device_to_img(Rgb *device_img, cv::Mat& img)
{
    int width = img.rows;
    int height = img.cols;
    std::cout << width * height << std::endl;

    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
        {
            img.at<cv::Vec3b>(j, i)[0] = (int)device_img[j + i * width].r;
            img.at<cv::Vec3b>(j, i)[1] = (int)device_img[j + i * width].g;
            img.at<cv::Vec3b>(j, i)[2] = (int)device_img[j + i * width].b;

        }
}

Rgb *empty_img_device(cv::Mat img)
{
    Rgb *device_img;
    int width = img.rows;
    int height = img.cols;
    cudaMallocManaged(&device_img, width * height * sizeof (Rgb));

    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
            device_img[j + i * width] = Rgb(0.0, 0.0, 0.0);

    return device_img;
}

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cout << "usage: main <Image_Path> <Conv_size>" << std::endl;
        return 1;
    }
    cv::Mat image;
    image = cv::imread(argv[1], CV_LOAD_IMAGE_UNCHANGED);
    if (!image.data)
    {
        std::cout << "Could not open or find the image" << std::endl;
        return 1;
    }


    Rgb* device_dst = empty_img_device(image);
    Rgb* device_img = img_to_device(image);


    kernel_conv<<<1, 1>>>(device_dst, device_img, image.rows, image.cols, std::stoi(argv[2]));

    cudaDeviceSynchronize();

    device_to_img(device_dst, image);

    cudaFree(device_dst);
    cudaFree(device_img);

    cv::namedWindow("Display Window", CV_WINDOW_AUTOSIZE);
    cv::imshow("Display Window", image);
    cv::waitKey(0);
    return 0;
}
