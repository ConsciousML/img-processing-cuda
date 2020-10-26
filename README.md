# Image Processing with CUDA C++

## Objective
The objective of this project is to implement from scratch in CUDA C++ various image processing algorithms.
A Cpu and a Gpu version of the following algorithms is implemented and commented:
- Canny Edge Detection
- Non Local-Means De-Noising
- K-Nearest Neighbors De-Noising
- Convolution Blurring
- Pixelize

We benchmarked the [Gpu](https://github.com/ConsciousML/canny-edge-cuda/blob/master/src/gpu/bench/benchs.ipynb) and [Cpu](https://github.com/ConsciousML/canny-edge-cuda/blob/master/src/cpu/bench/benchs.ipynb) version.

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
### Canny Edge Detection
![nlm](images/edge_detect.jpg)
<br>
Detects the edges of an image.
<br>

Usage:
```bash
    ./main <image_path> edge_detect
```

### Non Local-Means De-noising
![nlm](images/nlm_results.jpg)
<br>
Removes the grain of an image.
<br>
Benchmark:
- Cpu:

<img src="images/bench_cpu_nlm_514.png" height="256" width="414">

- Gpu:

<img src="images/bench_gpu_nlm_514.png" height="256" width="414">

Usage:
```bash
    ./main <image_path> nlm <conv_size> <hyper_param>
```

### K-Nearest Neighbors De-noising
![nlm](images/knn_results.jpg)
<br>
Removes the noise of an image using the KNN algorithm.
<br>
Benchmark:
- Cpu:

<img src="images/bench_cpu_knn.png" height="256" width="414">

- Gpu:

<img src="images/bench_gpu_knn.png" height="256" width="414">

Usage:
```bash
    ./main <image_path> nlm <conv_size> <block_radius> <weight_decay>
```

### Convolution Blurring
![conv_res](images/conv_res.jpg)
<br>
Blurs an image using the convolution operator.
<br>
Usage:
```bash
    ./main <image_path> conv <conv_size>
    ./main <image_path> shared_conv <conv_size>
```
Use `shared_conv` for an optimized version using shared memory.

### Pixelize
![conv_res](images/pixelize.jpg)
<br>
Pixelizes an image.
<br>
Usage
```bash
    ./main <image_path> pixelize <conv_size>
```
