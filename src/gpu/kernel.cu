#include <cstdio>
#include <cstdlib>
#include <valarray>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

__global__ void foo(int *a, int N) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;
    a[i]=i;
}

int test_kernel()
{ 
  int N=4097;
  int threads=128;
  int blocks=(N+threads-1)/threads;
  int *a;

  cudaMallocManaged(&a,N * sizeof(int));
  foo<<<blocks,threads>>>(a, N);
  cudaDeviceSynchronize();

  for(int i=0;i<10;i++)
    printf("%d\n",a[i]);

  return 0;

}

