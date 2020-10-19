# Image Processing with CUDA C++

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
Usage:
```bash
    ./main <image_path> nlm <conv_size> <hyper_param>
```

### K-Nearest Neighbor De-noising
![nlm](images/knn_results.jpg)
<br>
Removes the noise of an image using the KNN algorithm.
<br>
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
    ./main <image_path> conv <Conv_size>
    ./main <image_path> shared_conv <Conv_size>
```
Use `shared_conv` for an optimized version using shared memory.

### Pixelize
![conv_res](images/pixelize.jpg)
<br>
Pixelizes an image.
<br>
Usage
```bash
    ./main ../../../pictures/lenna.jpg pixelize <Conv_size>
```