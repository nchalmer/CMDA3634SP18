#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cuda.h"

__global__ void reduction(int N, float *a, float* result) {

  int thread = threadIdx.x;
  int block  = blockIdx.x;
  int blockSize = blockDim.x;

  //unique global thread ID
  int id = thread + block*blockSize;

  __shared__ float s_sum[32];

  s_sum[id] = a[id]; //add the thread's id to start
    
  __syncthreads(); //make sure the write to shared is finished
  
  if (id<16) {//first half
   s_sum[id] += s_sum[id+16]; 
  }
  
  __syncthreads(); //make sure the write to shared is finished

  if (id<8) {//next half
   s_sum[id] += s_sum[id+8]; 
  }
  
  __syncthreads(); //make sure the write to shared is finished

  if (id<4) {//next half
   s_sum[id] += s_sum[id+4]; 
  }
  
  __syncthreads(); //make sure the write to shared is finished

  if (id<2) {//next half
   s_sum[id] += s_sum[id+2]; 
  }
  
  __syncthreads(); //make sure the write to shared is finished

  if (id<1) {//final piece
    s_sum[id] += s_sum[id+1];
    *result = s_sum[id];
  }
}


//perform a reduction on a vector of length N
int main (int argc, char **argv) {
  
  int N = 32;

  double seed=0;
  srand48(seed);

  //allocate memory on host
  float *h_a = (float*) malloc(N*sizeof(float));
  
  //populate with random data
  for (int n=0;n<N;n++) {
    h_a[n] = drand48();
  }
  
  //perform the reduction on host
  float h_sum = 0.;
  for (int n=0;n<N;n++) {
    h_sum += h_a[n];
  }
  
  printf("The Host's sum was %f \n", h_sum);

  float *d_a, *d_sum;
  cudaMalloc(&d_a, N*sizeof(float));
  cudaMalloc(&d_sum, 1*sizeof(float));

  //populate the device array with the same data as the host
  cudaMemcpy(d_a,h_a,N*sizeof(float),cudaMemcpyHostToDevice);
  
  //block dimensions
  dim3 B(32,1,1);

  //grid dimensions
  dim3 G((N+32-1)/32,1,1);

  reduction <<< G,B >>> (N, d_a, d_sum);

  cudaMemcpy(&h_sum,d_sum,1*sizeof(float),cudaMemcpyDeviceToHost);

  printf("The Device's sum was %f \n", h_sum);

  return 0;
}
