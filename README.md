# Canny Edge Detection CUDA C++

## Presentation :
This project is about benchmarking CPU vs GPU using
a List of algorithms of Image processing.

## Setup:
```bash
    cd src/gpu
    mkdir build && cd build
    cmake ..
    make
```

## Algorithms :
### Non local-mean-algorithm
-https://en.wikipedia.org/wiki/Non-local_means
usage-cpu:
```bash
    ./main ../../../pictures/index.jpeg 3 1.2
    ./main <Image_path> <Conv_size> <Hyper_param>
```

### Convolution
for gpu convolution
usage-gpu:  
```bash
    ./main ../../../pictures/lenna.jpg conv <Conv_size>
```

for gpu SHARED convolution
usage-gpu: 
```bash
    ./main ../../../pictures/lenna.jpg shared_conv <Conv_size>
```

### Pixelize
for gpu SHARED pixelize
usage-gpu:  
```bash
    ./main ../../../pictures/lenna.jpg pixelize <Conv_size>
```