# CPU_vs_GPU_Image_Processing_Algorithms

## Presentation :
-the project is about benchmarking CPU vs GPU using
a List of algorithms of Image processing.

## Algorithms :
### Non local-mean-algorithm
-https://en.wikipedia.org/wiki/Non-local_means
usage-cpu:  cd src/cpu
            mkdir build && cd build
            cmake ..
            make
            ./main ../../../pictures/index.jpeg 3 1.2
            ./main <Image_path> <Conv_size> <Hyper_param>

### Convolution
for gpu convolution
usage-gpu:  cd src/gpu
            mkdir build && cd build
            cmake ..
            make
            ./main ../../../pictures/lenna.jpg conv <Conv_size>

for gpu SHARED convolution
usage-gpu:  cd src/gpu
            mkdir build && cd build
            cmake ..
            make
            ./main ../../../pictures/lenna.jpg shared_conv <Conv_size>

### Pixelize
for gpu SHARED pixelize
usage-gpu:  cd src/gpu
            mkdir build && cd build
            cmake ..
            make
            ./main ../../../pictures/lenna.jpg pixelize <Conv_size>

### fft algorithm
-https://en.wikipedia.org/wiki/fast_fourier_transform

