#include <stdio.h>
#include <iostream>
#include <valarray>
#include <assert.h>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#define TILE_WIDTH 16

struct Rgb {
    __host__ __device__ Rgb() {}
    __host__ __device__ Rgb(float x, float y, float z) : r(x), g(y), b(z) {}
    __host__ __device__ Rgb(cv::Vec3b v) : r(v[0]), g(v[1]), b(v[2]) {}
    __host__ __device__ void div(int x)
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
    cnt = 0;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= rows || y >= cols)
        return;

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

__global__ void kernel_conv_shared(Rgb* device_img, Rgb* img, int rows, int cols, int conv_size)
{
    __shared__ Rgb ds_img[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int dimx = blockDim.x;
    int dimy = blockDim.y;
    int x = bx * dimx + tx;
    int y = by * dimy + ty;

    for (int u = 0; u < rows / TILE_WIDTH + 1; u++)
    {
        for (int v = 0; v < cols / TILE_WIDTH + 1; v++)
        {
            if (u == bx && v == by)
            {
                auto elt = img[x + y * rows];
                ds_img[ty][tx] = Rgb(elt.r, elt.g, elt.b);
            }
            __syncthreads();
            auto elt = ds_img[ty][tx];
            device_img[x + y * rows].r = elt.r;
            device_img[x + y * rows].g = elt.g;
            device_img[x + y * rows].b = elt.b;
            __syncthreads();
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

    int width = image.rows;
    int height = image.cols;

    Rgb* device_dst = empty_img_device(image);
    Rgb* device_img = img_to_device(image);
    Rgb* out = (Rgb*)malloc(width * height * sizeof (Rgb));

    dim3 blockSize = dim3(16, 16);
    int bx = (width + blockSize.x - 1) / blockSize.x;
    int by = (height + blockSize.y - 1) / blockSize.y;
    dim3 gridSize = dim3(bx, by);

    kernel_conv_shared<<<gridSize, blockSize>>>(device_dst, device_img, width, height, std::stoi(argv[2]));

    cudaDeviceSynchronize();
    cudaMemcpy(out, device_dst, height * width * sizeof (Rgb), cudaMemcpyDeviceToHost);

    device_to_img(out, image);

    cudaFree(device_dst);
    cudaFree(device_img);
    free(out);

    cv::namedWindow("Display Window", CV_WINDOW_AUTOSIZE);
    cv::imshow("Display Window", image);
    cv::waitKey(0);
    return 0;
}
