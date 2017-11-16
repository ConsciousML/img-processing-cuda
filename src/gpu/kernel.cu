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

__device__ void gauss_conv(Rgb *image, Rgb& res, int x, int y, int width, int height, int conv_size, double h_param)
{
    auto cnt = Rgb(0.0, 0.0, 0.0);
    for (int j = y - conv_size; j < y + conv_size; j++)
    {
        for (int i = x - conv_size; i < x + conv_size; i++)
        {
            if (i >= 0 and j >= 0 and i < width and j < height)
            {
                auto ux = image[y * width + x];
                auto uy = image[j * width + i];
                double c1 = std::exp(-(std::pow(std::abs(i + j - (x + y)), 2)) / (double)std::pow(conv_size, 2));
                double h_div = std::pow(h_param, 2);

                auto c2 = Rgb(std::exp(-(std::pow(std::abs(uy.r - ux.r), 2)) / h_div),
                    std::exp(-(std::pow(std::abs(uy.g - ux.g), 2)) / h_div),
                    std::exp(-(std::pow(std::abs(uy.b - ux.b), 2)) / h_div));

                res.r += uy.r * c1 * c2.r;
                res.g += uy.g * c1 * c2.g;
                res.b += uy.b * c1 * c2.b;

                cnt.r += c1 * c2.r;
                cnt.g += c1 * c2.g;
                cnt.b += c1 * c2.b;
            }
        }
    }
    if (cnt.r != 0 and cnt.g != 0 and cnt.b != 0)
    {
        res.r /= cnt.r;
        res.g /= cnt.g;
        res.b /= cnt.b;
    }
}

__global__ void knn(Rgb* device_img, Rgb* img, int width, int height, int conv_size, double h_param)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width or y >= height)
        return;
    /*auto res = Rgb(0.0, 0.0, 0.0);
    gauss_conv(img, res, x, y, width, height, conv_size, h_param);
    device_img[y * width + x].r = res.r;
    device_img[y * width + x].g = res.g;
    device_img[y * width + x].b = res.b;*/
    //device_img[y * width + x] = img[y * width + x];
    device_img[y * width + x] = img[y * width + x];
}
__global__ void non_local_means_gpu(Rgb* device_img, Rgb* img, int conv_size, float weight_decay)
{

}
__global__ void kernel_shared_conv(Rgb* device_img, Rgb* img, int width, int height, int strel_size)
{
    int r = strel_size / 2;
    int block_w = TILE_WIDTH + 2 * r;
    extern __shared__ Rgb fast_acc_mat[];
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int row_o = blockIdx.y * TILE_WIDTH + ty;
    int col_o = blockIdx.x * TILE_HEIGHT + tx;
    int row_i = row_o - r;
    int col_i = col_o - r;

    if ((row_i >= 0) && (row_i < height) && (col_i >= 0) && (col_i < width))
    {
        auto elt = img[row_i * width + col_i];
        fast_acc_mat[ty * block_w + tx] = Rgb(elt.r, elt.g, elt.b);
    }
    else
        fast_acc_mat[ty * block_w + tx] = Rgb(0, 0, 0);
    __syncthreads();

    if (ty < TILE_HEIGHT && tx < TILE_WIDTH)
    {
        auto sum = Rgb(0, 0, 0);
        int cnt = 0;
        for (int i = 0; i < strel_size; i++)
        {
            for (int j = 0; j < strel_size; j++)
            {
                cnt++;
                sum.r += fast_acc_mat[(i + ty) * block_w + j + tx].r;
                sum.g += fast_acc_mat[(i + ty) * block_w + j + tx].g;
                sum.b += fast_acc_mat[(i + ty) * block_w + j + tx].b;
            }
        }
        if (row_o < height && col_o < width)
        {
            sum.r /= cnt;
            sum.g /= cnt;
            sum.b /= cnt;
            device_img[row_o * width + col_o] = sum;
        }
    }
}
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

__global__ void kernel_pixelize(Rgb* device_img, Rgb* img, int rows, int cols, int pix_size)
{
    extern __shared__ Rgb ds_img[];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int dimx = blockDim.x;
    int dimy = blockDim.y;
    int x = bx * dimx + tx;
    int y = by * dimy + ty;
    int cnt = 0;

    if (x >= rows || y >= cols)
        return;

    for (int u = 0; u < rows / pix_size + 1; u++)
    {
        for (int v = 0; v < cols / pix_size + 1; v++)
        {
            if (u == bx and v == by)
            {
                auto elt = img[x + y * rows];
                ds_img[ty * pix_size + tx] = Rgb(elt.r, elt.g, elt.b);
                __syncthreads();

                for (int i = y - pix_size; i < y + pix_size && i < cols; i++)
                {
                    for (int j = x - pix_size; j < x + pix_size && j < rows; j++)
                    {
                        if (i >= 0 and j >= 0
                                and i >= v * pix_size
                                and i < (v + 1) * pix_size
                                and j >= u * pix_size
                                and j < (u + 1) * pix_size)
                        {
                            cnt++;
                            int ds_x = j - u * pix_size;
                            int ds_y = i - v * pix_size;
                            auto elt = ds_img[ds_y * pix_size + ds_x];
                            device_img[x + y * rows].r += elt.r;
                            device_img[x + y * rows].g += elt.g;
                            device_img[x + y * rows].b += elt.b;
                        }
                    }
                }
                __syncthreads();
            }
        }
    }
    device_img[x + y * rows].r /= cnt;
    device_img[x + y * rows].g /= cnt;
    device_img[x + y * rows].b /= cnt;
}

