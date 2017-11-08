#include <stdio.h>
#include <iostream>
#include <valarray>
#include <assert.h>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "kernel.cuh"
#define TILE_WIDTH_PIX 4
#define TILE_WIDTH 16
#define TILE_HEIGHT 16
#define STREL_SIZE 5
#define R (STREL_SIZE / 2)
#define BLOCK_W (TILE_WIDTH + (2 * R))
#define BLOCK_H (TILE_HEIGHT + (2 * R))

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        std::cout << "usage: main <Image_Path> <Func_name> <Conv_size>" << std::endl;
        return 1;
    }
    cv::Mat image;
    image = cv::imread(argv[1], CV_LOAD_IMAGE_UNCHANGED);
    if (!image.data)
    {
        std::cout << "Could not open or find the image" << std::endl;
        return 1;
    }

    std::string func_name = argv[2];
    int width = image.rows;
    int height = image.cols;

    Rgb* device_dst = empty_img_device(image);
    Rgb* device_img = img_to_device(image);
    Rgb* out = (Rgb*)malloc(width * height * sizeof (Rgb));

    dim3 blockSize;
    if (func_name == "pixelize")
        blockSize = dim3(TILE_WIDTH_PIX, TILE_WIDTH_PIX);
    else
        blockSize = dim3(TILE_WIDTH, TILE_WIDTH);
    int bx = (width + blockSize.x - 1) / blockSize.x;
    int by = (height + blockSize.y - 1) / blockSize.y;
    dim3 gridSize = dim3(bx, by);

    if (func_name == "pixelize")
        kernel_pixelize<<<gridSize, blockSize>>>(device_dst, device_img, width, height, std::stoi(argv[3]));
    else if (func_name == "conv")
        kernel_conv<<<gridSize, blockSize>>>(device_dst, device_img, width, height, std::stoi(argv[3]));
    else if (func_name == "shared_conv")
    {
        dim3 block(16 + STREL_SIZE - 1, 16 + STREL_SIZE - 1);
        dim3 grid(width / (block.x) + block.x, height / (block.y) + block.y);
        kernel_shared_conv<<<grid, block>>>(device_dst, device_img, width, height);
    }
    else
    {
        std::cout << "error: function name '" << func_name << "' is not known." << std::endl;
        cudaFree(device_dst);
        cudaFree(device_img);
        free(out);
        return 1;
    }

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
