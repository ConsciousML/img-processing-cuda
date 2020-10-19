#include <stdio.h>
#include <iostream>
#include <valarray>
#include <assert.h>
#include <math.h>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "kernel.cuh"

#define TILE_WIDTH 16
#define TILE_HEIGHT 16


__device__ int mask1[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
__device__ int mask2[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};

// Kernel function for the canny edge detector
// Computes the gradient of edges and fixes broken edges
__global__ void hysterysis(Rgb *device_img, int* changed, int width, int height, double t)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width or y >= height)
        return;
    if (device_img[x + y * width].r == 255)
        return;

    double curr_dir = device_img[x + y * width].b;
    double curr_grad = device_img[x + y * width].g;
    changed[0] = 0;
    if (22.5 <= curr_dir and curr_dir < 67.5
            and (x - 1) >= 0
            and (y + 1) < height
            and (x + 1) < width
            and (y - 1) >= 0)
    {
        double dir1 = device_img[(x - 1) + (y + 1) * width].b;
        double dir2 = device_img[(x + 1) + (y - 1) * width].b;
        if (((22.5 <= dir1 and dir1 < 67.5) or (22.5 <= dir2 and dir2 < 67.5)) and curr_grad > t)
        {
            device_img[x + y * width].r = 255;
            changed[0] = 1;
        }
    }
    else if (67.5 <= curr_dir and curr_dir < 112.5
            and (x - 1) >= 0
            and (x + 1) < width)
    {
        double dir1 = device_img[(x - 1) + y * width].b;
        double dir2 = device_img[(x + 1) + y * width].b;
        if (((67.5 <= dir1 and dir1 < 112.5) or (67.5 < dir2 and dir2 < 112.5)) and curr_grad > t)
        {
            device_img[x + y * width].r = 255;
            changed[0] = 1;
        }
    }
    else if (112.5 <= curr_dir and curr_dir < 157.5
            and (x - 1) >= 0
            and (y - 1) >= 0
            and (x + 1) < width
            and (y + 1) < height)
    {
        double dir1 = device_img[(x - 1) + (y - 1) * width].b;
        double dir2 = device_img[(x + 1) + (y + 1) * width].b;
        if (((112.5 <= dir1 and dir1 < 157.5) or (112.5 <= dir2 and dir2 < 157.5)) and curr_grad > t)
        {
            device_img[x + y * width].r = 255;
            changed[0] = 1;
        }
    }
    else if (((0 <= curr_dir and curr_dir < 22.5)
                or (157.5 <= curr_dir and curr_dir <= 180.0))
            and (y - 1) >= 0
            and (y + 1) < height)
    {
        double dir1 = device_img[x + (y - 1) * width].b;
        double dir2 = device_img[x + (y + 1) * width].b;
        if ((((0 <= dir1 and dir1 < 22.5) and (157.5 <= dir1 and dir1 <= 180.5))
                    or ((0 <= dir2 and dir2 < 22.5) and (157.5 <= dir2 and dir2 <= 180.5))) and curr_grad > t)
        {
            device_img[x + y * width].r = 255;
            changed[0] = 1;
        }
    }
}

// Classifies the gradient direction for each edges
__global__ void non_max_suppr(Rgb *device_img, double* img, int width, int height, double thresh)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width or y >= height)
        return;
    double curr_dir = device_img[x + y * width].b;
    double curr_grad = device_img[x + y * width].g;
    if (22.5 <= curr_dir and curr_dir < 67.5
            and (x - 1) >= 0
            and (y + 1) < height
            and (x + 1) < width
            and (y - 1) >= 0)
    {
        double val1 = device_img[(x - 1) + (y + 1) * width].g;
        double val2 = device_img[(x + 1) + (y - 1) * width].g;
        if (val1 < curr_grad and val2 < curr_grad and curr_grad > thresh)
            device_img[x + y * width].r = 255;
        else
            device_img[x + y * width].r = 0;
    }
    else if (67.5 <= curr_dir and curr_dir < 112.5
            and (x - 1) >= 0
            and (x + 1) < width)
    {
        double val1 = device_img[(x - 1) + y * width].g;
        double val2 = device_img[(x + 1) + y * width].g;
        if (val1 < curr_grad and val2 < curr_grad and curr_grad > thresh)
            device_img[x + y * width].r = 255;
        else
            device_img[x + y * width].r = 0;
    }
    else if (112.5 <= curr_dir and curr_dir < 157.5
            and (x - 1) >= 0
            and (y - 1) >= 0
            and (x + 1) < width
            and (y + 1) < height)
    {
        double val1 = device_img[(x - 1) + (y - 1) * width].g;
        double val2 = device_img[(x + 1) + (y + 1) * width].g;
        if (val1 < curr_grad and val2 < curr_grad and curr_grad > thresh)
            device_img[x + y * width].r = 255;
        else
            device_img[x + y * width].r = 0;
    }
    else if (((0 <= curr_dir and curr_dir < 22.5)
                or (157.5 <= curr_dir and curr_dir <= 180.0))
            and (y - 1) >= 0
            and (y + 1) < height)
    {
        double val1 = device_img[x + (y - 1) * width].g;
        double val2 = device_img[x + (y + 1) * width].g;
        if (val1 < curr_grad and val2 < curr_grad and curr_grad > thresh)
            device_img[x + y * width].r = 255;
        else
            device_img[x + y * width].r = 0;
    }
}

// Computes a sobel convolution and compute the gradient direction
__global__ void sobel_conv(Rgb *device_img, double* img, int width, int height, int conv_size)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width or y >= height)
        return;
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
            if (i >= 0 and j >= 0 and i < width and j < height)
            {
                int weight1 = mask1[u][v];
                int weight2 = mask2[u][v];
                auto pix = img[i + j * width];
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
    double g = std::sqrt(std::pow(sum1, 2) + std::pow(sum2, 2));
    double d = atan2(sum2, sum1);
    d = (d > 0 ? d : (2 * M_PI + d)) * 360 / (2 * M_PI);
    device_img[x + y * width].g = g;
    device_img[x + y * width].b = d;

}

// Applies the convolution operator on the image
__device__ void conv(Rgb *image, Rgb& rgb, int width, int height, int x1, int y1, int x2, int y2, int conv_size)
{
    int cnt = 0;
    for (int j1 = y1 - conv_size; j1 < y1 + conv_size; j1++)
        for (int i1 = x1 - conv_size; i1 < x1 + conv_size; i1++)
        {
            int i2 = i1 - x1 + x2;
            int j2 = j1 - y1 + y2;
            if (i1 >= 0 and j1 >= 0 and
                    j2 >= 0 and i2 >= 0 and
                    i1 < height and j1 < width and
                    i2 < height and j2 < width)
            {
                cnt++;
                auto pix1 = image[i1 * width + j1];
                auto pix2 = image[i2 * width + j2];
                rgb.r += std::pow(std::abs(pix1.r - pix2.r), 2);
                rgb.g += std::pow(std::abs(pix1.g - pix2.g), 2);
                rgb.b += std::pow(std::abs(pix1.b - pix2.b), 2);
            }
        }
    if (cnt > 0) {
        rgb.r /= cnt;
        rgb.g /= cnt;
        rgb.b /= cnt;
    }
}

// Applies a gaussian mask to the image
__device__ void gauss_conv_nlm(Rgb *image, Rgb& res, int x, int y, int width, int height, int conv_size, int block_radius, double h_param)
{
    auto cnt = Rgb(0.0, 0.0, 0.0);
    for (int j = y - conv_size; j < y + conv_size; j++)
    {
        for (int i = x - conv_size; i < x + conv_size; i++)
        {
            if (i >= 0 and j >= 0 and i < width and j < height)
            {
                auto u = Rgb(0, 0, 0);
                conv(image, u, width, height, y, x, j, i, block_radius);
                auto uy = image[j * width + i];
                double c1 = std::exp(-(std::pow(std::abs(i + j - (x + y)), 2)) / (double)std::pow(conv_size, 2));
                double h_div = std::pow(h_param, 2);

                auto c2 = Rgb(std::exp(-u.r / h_div),
                        std::exp(-u.g / h_div),
                        std::exp(-u.b / h_div));

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

// Non Local-Means kernel function
__global__ void nlm(Rgb* device_img, Rgb* img, int width, int height, int conv_size, int block_radius, double h_param)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width or y >= height)
        return;
    auto res = Rgb(0.0, 0.0, 0.0);
    gauss_conv_nlm(img, res, x, y, width, height, conv_size, block_radius, h_param);
    device_img[y * width + x].r = res.r;
    device_img[y * width + x].g = res.g;
    device_img[y * width + x].b = res.b;
}

// Shared knn kernel function
// Uses the K-Nearest Neighbors algorithm for de-noising an image
// Uses the shared memory optimization
__global__ void shared_knn(Rgb* device_img, Rgb* img, int width, int height, int strel_size, double h_param)
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

    if (row_o >= 0 and col_o >= 0 and row_o < height and col_o < width)
    {
        if (ty < TILE_HEIGHT && tx < TILE_WIDTH)
        {
            auto sum = Rgb(0, 0, 0);
            auto cnt = Rgb(0, 0, 0);
            auto ux = img[row_o * width + col_o];
            for (int i = 0; i < strel_size; i++)
            {
                for (int j = 0; j < strel_size; j++)
                {
                    auto uy = fast_acc_mat[(i + ty) * block_w + j + tx];
                    double h_div = std::pow(h_param, 2);
                    double c1 = std::exp(-(std::pow(std::abs(col_i + row_i + i + j - (row_o + col_o)), 2)) / (double)std::pow(r, 2));
                    auto c2 = Rgb(std::exp(-(std::pow(std::abs(uy.r - ux.r), 2)) / h_div),
                            std::exp(-(std::pow(std::abs(uy.g - ux.g), 2)) / h_div),
                            std::exp(-(std::pow(std::abs(uy.b - ux.b), 2)) / h_div));

                    sum.r += uy.r * c1 * c2.r;
                    sum.g += uy.g * c1 * c2.g;
                    sum.b += uy.b * c1 * c2.b;

                    cnt.r += c1 * c2.r;
                    cnt.g += c1 * c2.g;
                    cnt.b += c1 * c2.b;
                }
            }
            sum.r /= cnt.r;
            sum.g /= cnt.g;
            sum.b /= cnt.b;
            device_img[row_o * width + col_o] = sum;
        }
    }
}

// Shared knn kernel function
// Uses the K-Nearest Neighbors algorithm for de-noising an image
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

// Hat function of the K-Nearest Neigbhors algorithm
__global__ void knn(Rgb* device_img, Rgb* img, int width, int height, int conv_size, double h_param)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width or y >= height)
        return;
    auto res = Rgb(0.0, 0.0, 0.0);
    gauss_conv(img, res, x, y, width, height, conv_size, h_param);
    device_img[y * width + x].r = res.r;
    device_img[y * width + x].g = res.g;
    device_img[y * width + x].b = res.b;
}

// Hat function of the K-Nearest Neigbhors algorithm
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

// Hat function for the convolution algorithm
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

// Pixelizes an image
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

