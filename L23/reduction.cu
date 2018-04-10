#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cuda.h"

__global__ void reduction(const int N, float *a, float *result) {

  int thread = threadIdx.x;
  int block  = blockIdx.x;
  int blockSize = blockDim.x;

  //unique global thread ID
  int id = thread + block*blockSize;

  __shared__ float s_sum[256];

  float sum = 0;
  if (id<N) 
    sum = a[id]; //add the thread's id to start
  
  s_sum[thread] = sum;
  
  __syncthreads(); //make sure the write to shared is finished
   
  if (thread<128) {//first half
    s_sum[thread] += s_sum[thread+128]; 
  }
  
  __syncthreads(); //make sure the write to shared is finished

 
  if (thread<64) {//next half
    s_sum[thread] += s_sum[thread+64]; 
  }
  
  __syncthreads(); //make sure the write to shared is finished

  if (thread<32) {//next half
    s_sum[thread] += s_sum[thread+32]; 
  }
  
  __syncthreads(); //make sure the write to shared is finished

  if (thread<16) {//next half
    s_sum[thread] += s_sum[thread+16]; 
  }
  
  __syncthreads(); //make sure the write to shared is finished

  if (thread<8) {//next half
   s_sum[thread] += s_sum[thread+8]; 
  }
  
  __syncthreads(); //make sure the write to shared is finished

  if (thread<4) {//next half
   s_sum[thread] += s_sum[thread+4]; 
  }
  
  __syncthreads(); //make sure the write to shared is finished

  if (thread<2) {//next half
   s_sum[thread] += s_sum[thread+2]; 
  }
  
  __syncthreads(); //make sure the write to shared is finished

  if (thread<1) {//final piece
    s_sum[thread] += s_sum[thread+1];
    result[block] = s_sum[thread];
  }
}


//perform a reduction on a vector of length N
int main (int argc, char **argv) {
  
  int N = atoi(argv[1]);

  double seed=clock();
  srand48(seed);

  //allocate memory on host
  float *h_a = (float*) malloc(N*sizeof(float));
  
  //populate with random data
  for (int n=0;n<N;n++) {
    h_a[n] = drand48();
  }
  
  //perform the reduction on host
  float sum = 0.;
  for (int n=0;n<N;n++) {
    sum += h_a[n];
  }
  
  printf("The Host's sum was %f \n", sum);

  float *d_a, *d_sum;

  int Nnew = (N+256-1)/256; //size of the 'reduced' vector

  cudaMalloc(&d_a, N*sizeof(float));
  cudaMalloc(&d_sum, Nnew*sizeof(float));

  float *h_sum = (float*) malloc(Nnew*sizeof(float));

  //populate the device array with the same data as the host
  cudaMemcpy(d_a,h_a,N*sizeof(float),cudaMemcpyHostToDevice);
  
  do {
 
    Nnew = (N+256-1)/256;

    //block dimensions
    dim3 B(256,1,1);

    //grid dimensions
    dim3 G(Nnew,1,1);
    
    reduction <<< G,B >>> (N, d_a, d_sum);  
    
    //overwrite the a vector with the partially reduced vector
    cudaMemcpy(d_a,d_sum,Nnew*sizeof(float),cudaMemcpyDeviceToDevice);

    N = Nnew;
  } while (Nnew>1);
  
  cudaMemcpy(h_sum,d_sum,1*sizeof(float),cudaMemcpyDeviceToHost);

  printf("The Device's sum was %f \n", h_sum[0]);

  free(h_sum);
  free(h_a);
  cudaFree(d_a);
  cudaFree(d_sum);

  return 0;
}
