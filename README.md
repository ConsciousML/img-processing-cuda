# Canny Edge Detection CUDA C++

## Presentation :
This project is about benchmarking CPU vs GPU using
a List of algorithms of Image processing.

## Setup:
Make sure you have a CUDA capable GPU and install cudatoolkit for your OS.
Then run:
```bash
    cd src/gpu
    mkdir build && cd build
    cmake ..
    make
```

## Algorithms :
### Non local-mean-algorithm
Removes the grain of an image.
Usage:
```bash
    ./main <image_path> nlm <conv_size> <hyper_param>
```

### Convolution Blurring
Blurs an image using the convolution operator.
Usage:
```bash
    ./main <image_path> conv <Conv_size>
    ./main <image_path> shared_conv <Conv_size>
```
Use `shared_conv` for an optimized version using shared memory.

### Pixelize
for gpu SHARED pixelize
usage-gpu:  
```bash
    ./main ../../../pictures/lenna.jpg pixelize <Conv_size>
```