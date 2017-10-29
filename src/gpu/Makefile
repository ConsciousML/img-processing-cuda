SRC = main.cpp kernel.cu
CFLAGS = -lcuda -lopencv_gpu -lopencv_core -lopencv_highgui
all:
	nvcc $(SRC) -o main $(CFLAGS)
clean:
	rm -f main *.o
