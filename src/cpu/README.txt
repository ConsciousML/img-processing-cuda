to compile the project run the following commands:
mkdir build
cd build
cmake..
make

usage: ./main <Image_Path> <Func_name> <Func_param>

algorithms:
    -nlm:
        usage: ./main <Image_path> nlm <Conv_size> <Block_radius> <Weight_decay_param>
        example: ./main ../../../pictures/lenna.jpg nlm 2 2 150.0

    -knn:
        usage: ./main <Image_path> knn <Conv_size> <Weight_decay_param>
        example: ./main ../../../pictures/lenna.jpg knn 2 150.0

    -conv:
        usage: ./main <Image_path> conv <Conv_size>
        example: ./main ../../../pictures/lenna.jpg conv 2

    -edge_detect:
        usage: ./main <Image_path> edge_detect
        example: ./main ../../../pictures/lenna.jpg edge_detect
