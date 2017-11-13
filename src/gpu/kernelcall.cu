#include <stdio.h>
#include <iostream>
#include <valarray>
#include <assert.h>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "kernel.cuh"

#define TILE_WIDTH 16
#define TILE_HEIGHT 16

#define STREL_SIZE 5
#define R (STREL_SIZE / 2)
#define BLOCK_W (TILE_WIDTH + (2 * R))
#define BLOCK_H (TILE_HEIGHT + (2 * R))

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

void kernel_shared_conv_host(Rgb* device_img, Rgb* img, int width, int height, int conv_size)
{
    dim3 blockSize = dim3(TILE_WIDTH + STREL_SIZE - 1, TILE_WIDTH + STREL_SIZE - 1);
    int bx = width / (blockSize.x) + blockSize.x;
    int by = height / (blockSize.y) + blockSize.y;
    dim3 gridSize = dim3(bx, by);
    kernel_shared_conv<<<gridSize, blockSize>>>(device_img, img, width, height, conv_size);
}

void kernel_conv_host(Rgb* device_img, Rgb* img, int width, int height, int conv_size)
{
    dim3 blockSize = dim3(TILE_WIDTH, TILE_WIDTH);
    int bx = (width + blockSize.x - 1) / blockSize.x;
    int by = (height + blockSize.y - 1) / blockSize.y;
    dim3 gridSize = dim3(bx, by);
    kernel_conv<<<gridSize, blockSize>>>(device_img, img, width, height, conv_size);
}

void kernel_pixelize_host(Rgb* device_img, Rgb* img, int width, int height, int pix_size)
{
    dim3 blockSize = dim3(pix_size, pix_size);
    int bx = (width + blockSize.x - 1) / blockSize.x;
    int by = (height + blockSize.y - 1) / blockSize.y;
    dim3 gridSize = dim3(bx, by);
    kernel_pixelize<<<gridSize, blockSize, pix_size * pix_size * sizeof (Rgb)>>>(device_img, img, width, height, pix_size);
}

void kernel_non_local_means_host(Rgb* device_img, Rgb* img, int width, int height, int conv_size, float weight_decay)
{
    dim3 blockSize = dim3(TILE_WIDTH + STREL_SIZE - 1, TILE_WIDTH + STREL_SIZE - 1);
    int bx = width / (blockSize.x) + blockSize.x;
    int by = height / (blockSize.y) + blockSize.y;
    dim3 gridSize = dim3(bx, by);
    non_local_means_gpu<<<gridSize, blockSize>>>(device_img, img, conv_size, weight_decay);
}
