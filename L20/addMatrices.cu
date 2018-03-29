#include <stdio.h> 
#include <stdlib.h>
#include <time.h> 
#include <math.h>

#include "cuda.h"

__global__ void kernelAddMatrices1D(int N, double *A, double *B, double *C) {

  int threadId = threadIdx.x;
  int blockId = blockIdx.x;
  int blockSize = blockDim.x; //32

  int id = threadId + blockId*blockSize;

  C[id] = A[id] + B[id];

}

__global__ void kernelAddMatrices2D(int N, double *A, double* B, double *C) {

  int tIdx = threadIdx.x;
  int tIdy = threadIdx.y;

  int bIdx = blockIdx.x;
  int bIdy = blockIdx.y;

  int bSizex = blockDim.x;
  int bSizey = blockDim.y;

  
  int i = tIdx + bIdx*bSizex; //unique x coordinate
  int j = tIdy + bIdy*bSizey; //unique y coordinate

  int nx = 1024;
  C[i+j*nx] = A[i+j*nx] + B[i+j*nx]; 

}

__global__ void kernelMatrixTranspose2D_v1(double *A,  double *AT) {

  int tIdx = threadIdx.x;
  int tIdy = threadIdx.y;

  int bIdx = blockIdx.x;
  int bIdy = blockIdx.y;

  int bSizex = blockDim.x;
  int bSizey = blockDim.y;

  int i = tIdx + bIdx*bSizex; //unique x coordinate
  int j = tIdy + bIdy*bSizey; //unique y coordinate

  int nx = 1024;
  AT[i+j*nx] = A[j+i*nx]; 

}

//do the transpose using shared memory to get better device memory acceses
__global__ void kernelMatrixTranspose2D_v2(double *A,  double *AT) {

  int tIdx = threadIdx.x;
  int tIdy = threadIdx.y;

  int bIdx = blockIdx.x;
  int bIdy = blockIdx.y;

  int bSizex = blockDim.x;
  int bSizey = blockDim.y;

  __shared__ double s_A[32][32];

  int i = tIdx + bIdx*bSizex; //unique x coordinate
  int j = tIdy + bIdy*bSizey; //unique y coordinate

  int nx = 1024;

  //fetch a block of A into the shared array s_A
  s_A[tIdx][tIdy] = A[i+j*nx]; //read from A and write the block's transpose

  __syncthreads(); // barrier the threads on this block so all the 
                    // writes to s_A are completed
  
  AT[i+j*nx] = s_A[tIdy][tIdx]; // write out
}


int main(int argc, char **argv) {

  // dimensions of the matrices
  int nx = 1024;
  int ny = 1024;
  
  int N = nx*ny;

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
  for (int j=0;j<ny;j++) {
    for (int i=0;i<nx;i++) {
      int id = i+j*nx;
      h_c[id] = h_a[id] + h_b[id];  
    }
  }
  
  double hostEnd = clock();
  double hostTime = (hostEnd - hostStart)/(double) CLOCKS_PER_SEC;

  size_t inputMem = 2*N*sizeof(double); //number of bytes the operation inputs
  size_t outMem   = 1*N*sizeof(double); //number of bytes the operation outputs

  size_t totalMem = (inputMem+outMem);

  printf("The host took %f seconds to add a and b \n", hostTime);
  printf("The efective bandwidth of the host was: %f GB/s\n", totalMem/(1E9*hostTime));
  printf("\n");
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
  printf("\n");
  //at this point the data is allocated and populated on the device

  int Nthreads = 32; //get the number of threads per block from command line
  int Nblocks = (N+Nthreads-1)/Nthreads;

  double deviceStart = clock();
  kernelAddMatrices1D <<<Nblocks , Nthreads >>>(N, d_a, d_b, d_c);
  cudaDeviceSynchronize();  

  double deviceEnd = clock();
  double deviceTime = (deviceEnd-deviceStart)/(double) CLOCKS_PER_SEC;

  printf("The 1D Kernel took %f seconds to add a and b \n", deviceTime); 
  printf("The efective bandwidth of the 1D kernel was: %f GB/s\n", totalMem/(1E9*deviceTime));
  
  //use 2D thread blocks instead
  int Nthreadsx = 32;
  int Nthreadsy = 32;
  int Nthreadsz = 1;

  //declare the size of the block
  // Nthreadsx*Nthreadsy*Nthreadsz <= 1024
  dim3 Nthreads3(Nthreadsx,Nthreadsy,Nthreadsz); 
  
  //set number of blocks
  int Nblocksx = (nx+Nthreadsx-1)/Nthreadsx;   
  int Nblocksy = (ny+Nthreadsy-1)/Nthreadsy;   
  int Nblocksz = 1;   
  dim3 Nblocks3(Nblocksx,Nblocksy,Nblocksz); 

  
  deviceStart = clock();

  kernelAddMatrices2D <<<Nblocks3 , Nthreads3 >>>(N, d_a, d_b, d_c);
  
  cudaDeviceSynchronize();  
  deviceEnd = clock();
  deviceTime = (deviceEnd-deviceStart)/(double) CLOCKS_PER_SEC;

  printf("The 2D Kernel took %f seconds to add a and b \n", deviceTime); 
  printf("The efective bandwidth of the 2D kernel was: %f GB/s\n", totalMem/(1E9*deviceTime));
 


  printf("The device was %f times faster\n", hostTime/deviceTime);

  copyStart = clock();
  cudaMemcpy(h_c,d_c,N*sizeof(double),cudaMemcpyDeviceToHost);
  copyEnd = clock();
  copyTime = (copyEnd-copyStart)/(double) CLOCKS_PER_SEC;

  printf("It took %f seconds to copy the data back to the host. \n",copyTime);
  printf("The efective bandwidth of the copy was: %f GB/s\n", outMem/(1E9*copyTime));


   deviceStart = clock();

  // C = A^T
  kernelMatrixTranspose2D_v1 <<<Nblocks3 , Nthreads3 >>>(d_a, d_c);
  
  cudaDeviceSynchronize();  
  deviceEnd = clock();
  deviceTime = (deviceEnd-deviceStart)/(double) CLOCKS_PER_SEC;

  printf("The v1 tarnspose kernel took %f seconds to add a and b \n", deviceTime); 
  printf("The efective bandwidth of the v1 transpose kernel was: %f GB/s\n", totalMem/(1E9*deviceTime));
 
 

   deviceStart = clock();

  // C = A^T
  kernelMatrixTranspose2D_v2<<<Nblocks3 , Nthreads3 >>>(d_a, d_c);
  
  cudaDeviceSynchronize();  
  deviceEnd = clock();
  deviceTime = (deviceEnd-deviceStart)/(double) CLOCKS_PER_SEC;

  printf("The v2 tarnspose kernel took %f seconds to add a and b \n", deviceTime); 
  printf("The efective bandwidth of the v2 transpose kernel was: %f GB/s\n", totalMem/(1E9*deviceTime));
 



  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  free(h_a);
  free(h_b);
  free(h_c);
}
