#include <stdio.h> 
#include <stdlib.h>
#include <time.h> 
#include <math.h>

#include "cuda.h"

//device function
__global__ void kernelAddVectors(int N, double *a, double *b, double *c) {

  int threadid = threadIdx.x; //thread number
  int blockid = blockIdx.x; //block number
  int Nblock = blockDim.x;  //number of threads in a block

  int id = threadid + blockid*Nblock;

  if (id < N) {
    c[id] = a[id] + b[id]; 
  } 
}


int main(int argc, char **argv) {

  // get vector size from command line argument
  int N = atoi(argv[1]);

  //seed RNG
  double seed = clock();
  srand48(seed);

  double *h_a, *h_b, *h_c; //host vectors

  // allocate storage
  h_a = (double *) malloc(N*sizeof(double));
  h_b = (double *) malloc(N*sizeof(double));
  h_c = (double *) malloc(N*sizeof(double));

  //populate a and b
  for (int n=0;n<N;n++) {
    h_a[n] = drand48();
    h_b[n] = drand48();
  }

  double hostStart = clock();

  // c = a + b
  for (int n=0;n<N;n++) {
    h_c[n] = h_a[n] + h_b[n];
  }
  
  double hostEnd = clock();
  double hostTime = (hostEnd - hostStart)/(double) CLOCKS_PER_SEC;

  size_t inputMem = 2*N*sizeof(double); //number of bytes the operation inputs
  size_t outMem   = 1*N*sizeof(double); //number of bytes the operation outputs

  size_t totalMem = (inputMem+outMem);

  printf("The host took %f seconds to add a and b \n", hostTime);
  printf("The efective bandwidth of the host was: %f GB/s\n", totalMem/(1E9*hostTime));
  
  //Device arrays
  double *d_a, *d_b, *d_c;

  //allocate memory on the Device with cudaMalloc
  cudaMalloc(&d_a,N*sizeof(double));
  cudaMalloc(&d_b,N*sizeof(double));
  cudaMalloc(&d_c,N*sizeof(double));

  double copyStart = clock();

  //copy data from the host to the device
  cudaMemcpy(d_a,h_a,N*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(d_b,h_b,N*sizeof(double),cudaMemcpyHostToDevice);
  
  double copyEnd = clock();
  double copyTime = (copyEnd-copyStart)/(double)CLOCKS_PER_SEC;

  printf("It took %f seconds to copy the data to device. \n",copyTime);
  printf("The efective bandwidth of the copy was: %f GB/s\n", inputMem/(1E9*copyTime));

  //at this point the data is allocated and populated on the device

  int Nthreads = atoi(argv[2]); //get the number of threads per block from command line
  int Nblocks = (N+Nthreads-1)/Nthreads;

  double deviceStart = clock();

  kernelAddVectors <<<Nblocks ,Nthreads >>>(N, d_a, d_b, d_c);
  
  cudaDeviceSynchronize();  

  double deviceEnd = clock();
  double deviceTime = (deviceEnd-deviceStart)/(double) CLOCKS_PER_SEC;

  printf("The device took %f seconds to add a and b \n", deviceTime); 
  printf("The efective bandwidth of the device was: %f GB/s\n", totalMem/(1E9*deviceTime));
  printf("The device was %f times faster\n", hostTime/deviceTime);

  copyStart = clock();
  cudaMemcpy(h_c,d_c,N*sizeof(double),cudaMemcpyDeviceToHost);
  copyEnd = clock();
  copyTime = (copyEnd-copyStart)/(double) CLOCKS_PER_SEC;

  printf("It took %f seconds to copy the data back to the host. \n",copyTime);
  printf("The efective bandwidth of the copy was: %f GB/s\n", outMem/(1E9*copyTime));

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  free(h_a);
  free(h_b);
  free(h_c);
}
