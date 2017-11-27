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

//#define STREL_SIZE 5
//#define R (STREL_SIZE / 2)
//#define BLOCK_W (TILE_WIDTH + (2 * R))
//#define BLOCK_H (TILE_HEIGHT + (2 * R))

void device_to_img_grey(Rgb *device_img, cv::Mat& img)
{
    int width = img.rows;
    int height = img.cols;
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
            img.at<uchar>(j, i) = device_img[j + i * width].r;
}

void device_to_img(Rgb *device_img, cv::Mat& img)
{
    int width = img.rows;
    int height = img.cols;
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
        {
            img.at<cv::Vec3b>(j, i)[0] = device_img[j + i * width].r;
            img.at<cv::Vec3b>(j, i)[1] = device_img[j + i * width].g;
            img.at<cv::Vec3b>(j, i)[2] = device_img[j + i * width].b;

        }
}

double *img_to_device_grey(cv::Mat img)
{
    double *device_img;
    int width = img.rows;
    int height = img.cols;
    cudaMallocManaged(&device_img, width * height * sizeof (double));

    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
            device_img[j + i * width] = img.at<uchar>(j, i);

    return device_img;
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

double *empty_img_device_grey(cv::Mat img)
{
    double *device_img;
    int width = img.rows;
    int height = img.cols;
    cudaMallocManaged(&device_img, width * height * sizeof (double));

    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
            device_img[j + i * width] = 0.0;

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

void kernel_shared_conv_host(Rgb* device_img, Rgb* img, int width, int height, int r)
{
    int strel_size = 2 * r + r % 2;
    if (strel_size <= 0 or strel_size > 16)
    {
        std::cout << "\nerror: <Strel_size> parameter must be between 1 and 16 due to shared memory constaint.\n" << std::endl;
        assert(strel_size > 0 and strel_size < 16);
        return;
    }
    int block_w = TILE_WIDTH + 2 * r;
    dim3 blockSize = dim3(block_w, block_w);
    int bx = (width / TILE_WIDTH - 1) + blockSize.x;
    int by = (height / TILE_HEIGHT - 1) + blockSize.y;
    dim3 gridSize = dim3(bx, by);
    kernel_shared_conv<<<gridSize, blockSize, block_w * block_w * sizeof (Rgb)>>>(device_img, img, width, height, strel_size);
}

void kernel_conv_host(Rgb* device_img, Rgb* img, int width, int height, int conv_size)
{
    if (conv_size <= 0)
    {
        std::cout << "\nerror: <Conv_size> parameter must be strictly greater than 0.\n" << std::endl;
        assert(conv_size > 0);
        return;
    }
    dim3 blockSize = dim3(TILE_WIDTH, TILE_WIDTH);
    int bx = (width + blockSize.x - 1) / blockSize.x;
    int by = (height + blockSize.y - 1) / blockSize.y;
    dim3 gridSize = dim3(bx, by);
    kernel_conv<<<gridSize, blockSize>>>(device_img, img, width, height, conv_size);
}

void kernel_pixelize_host(Rgb* device_img, Rgb* img, int width, int height, int pix_size)
{
    if (pix_size <= 1 or pix_size > 32)
    {
        std::cout << "\nerror: <Pix_size> parameter must be between 2 and 32 included.\n" << std::endl;
        assert(pix_size > 1 and pix_size < 33);
        return;
    }
    dim3 blockSize = dim3(pix_size, pix_size);
    int bx = (width + blockSize.x - 1) / blockSize.x;
    int by = (height + blockSize.y - 1) / blockSize.y;
    dim3 gridSize = dim3(bx, by);
    kernel_pixelize<<<gridSize, blockSize, pix_size * pix_size * sizeof (Rgb)>>>(device_img, img, width, height, pix_size);
}

void kernel_knn_host(Rgb* device_img, Rgb* img, int width, int height, int conv_size, double h_param)
{
    dim3 blockSize = dim3(TILE_WIDTH, TILE_WIDTH);
    int bx = (width + blockSize.x - 1) / blockSize.x;
    int by = (height + blockSize.y - 1) / blockSize.y;
    dim3 gridSize = dim3(bx, by);
    knn<<<gridSize, blockSize>>>(device_img, img, width, height, conv_size, h_param);
}

void kernel_shared_knn_host(Rgb* device_img, Rgb* img, int width, int height, int r, double h_param)
{
    int strel_size = 2 * r + r % 2;
    if (strel_size <= 0 or strel_size > 16)
    {
        std::cout << "\nerror: <Strel_size> parameter must be between 1 and 16 due to shared memory constaint.\n" << std::endl;
        assert(strel_size > 0 and strel_size < 16);
        return;
    }
    int block_w = TILE_WIDTH + 2 * r;
    dim3 blockSize = dim3(block_w, block_w);
    int bx = (width / TILE_WIDTH - 1) + blockSize.x;
    int by = (height / TILE_HEIGHT - 1) + blockSize.y;
    dim3 gridSize = dim3(bx, by);
    shared_knn<<<gridSize, blockSize, block_w * block_w * sizeof (Rgb)>>>(device_img, img, width, height, strel_size, h_param);
}

void kernel_nlm_host(Rgb* device_img, Rgb* img, int width, int height, int conv_size, int block_radius, double h_param)
{
    dim3 blockSize = dim3(TILE_WIDTH, TILE_WIDTH);
    int bx = (width + blockSize.x - 1) / blockSize.x;
    int by = (height + blockSize.y - 1) / blockSize.y;
    dim3 gridSize = dim3(bx, by);
    nlm<<<gridSize, blockSize>>>(device_img, img, width, height, conv_size, block_radius, h_param);
}

void kernel_edge_detect(Rgb* device_img, double* img, int width, int height, int conv_size, double otsu_threshold)
{
    int mask1[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    int mask2[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    dim3 blockSize = dim3(TILE_WIDTH, TILE_WIDTH);
    int bx = (width + blockSize.x - 1) / blockSize.x;
    int by = (height + blockSize.y - 1) / blockSize.y;
    dim3 gridSize = dim3(bx, by);

    double *grad;
    double *dir;

    sobel_conv<<<gridSize, blockSize>>>(device_img, img, grad, dir, width, height, conv_size, mask1, mask2);

    /*cv::Mat dst;
    cv::Mat tmp_image;
    double otsu_threshold = cv::threshold(image, dst, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    auto edge_image = image.clone();
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
    return edge_image;*/
    delete(grad);
    delete(dir);
}


