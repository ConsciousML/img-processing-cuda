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

__global__ void non_local_means_gpu(Rgb* device_img, Rgb* img, int conv_size, float weight_decay)
{

}
__global__ void kernel_shared_conv(Rgb* device_img, Rgb* img, int width, int height, int conv_size)
{
    __shared__ Rgb fast_acc_mat[BLOCK_W][BLOCK_H];
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int row_o = blockIdx.y * TILE_WIDTH + ty;
    int col_o = blockIdx.x * TILE_HEIGHT + tx;
    int row_i = row_o - R;
    int col_i = col_o - R;

    if ((row_i >= 0) && (row_i < height) && (col_i >= 0) && (col_i < width))
    {
        auto elt = img[row_i * width + col_i];
        fast_acc_mat[ty][tx] = Rgb(elt.r, elt.g, elt.b);
    }
    else
        fast_acc_mat[ty][tx] = Rgb(0, 0, 0);
    __syncthreads();

    if (ty < TILE_HEIGHT && tx < TILE_WIDTH)
    {
        auto sum = Rgb(0, 0, 0);
        int cnt = 0;
        for (int i = 0; i < STREL_SIZE; i++)
        {
            for (int j = 0; j < STREL_SIZE; j++)
            {
                cnt++;
                sum.r += fast_acc_mat[i + ty][j + tx].r;
                sum.g += fast_acc_mat[i + ty][j + tx].g;
                sum.b += fast_acc_mat[i + ty][j + tx].b;
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

