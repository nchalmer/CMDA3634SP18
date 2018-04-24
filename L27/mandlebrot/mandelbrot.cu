/* 

To compile:

   nvcc -O3 -o mandelbrot mandelbrot.c png_util.c -I. -lpng -lm -fopenmp

Or just type:

   module load gcc
   make

To create an image with 4096 x 4096 pixels (last argument will be used to set number of threads):

    ./mandelbrot 4096 4096 1

*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "png_util.h"

// Q2a: add include for CUDA header file here:
#include "cuda.h"

#define MXITER 1000

typedef struct {
  
  double r;
  double i;
  
}complex_t;

// return iterations before z leaves mandelbrot set for given c
__device__ int testpoint(complex_t c){
  
  int iter;

  complex_t z;
  double temp;
  
  z = c;
  
  for(iter=0; iter<MXITER; iter++){
    
    temp = (z.r*z.r) - (z.i*z.i) + c.r;
    
    z.i = z.r*z.i*2. + c.i;
    z.r = temp;
    
    if((z.r*z.r+z.i*z.i)>4.0){
      return iter;
    }
  }
  
  
  return iter;
  
}

// perform Mandelbrot iteration on a grid of numbers in the complex plane
// record the  iteration counts in the count array

// Q2c: transform this function into a CUDA kernel
__global__ void  mandelbrot(int Nre, int Nim, complex_t cmin, complex_t cmax, float *count){ 

  int tIdx = threadIdx.x;
  int tIdy = threadIdx.y;

  int bIdx = blockIdx.x;
  int bIdy = blockIdx.y;

  int bSizex = blockDim.x;
  int bSizey = blockDim.y;

  int i = tIdx + bIdx*bSizex; //unique x coordinate
  int j = tIdy + bIdy*bSizey; //unique y coordinate

  complex_t c;

  double dr = (cmax.r-cmin.r)/(Nre-1);
  double di = (cmax.i-cmin.i)/(Nim-1);;


  c.r = cmin.r + dr*i;
  c.i = cmin.i + di*j;
    
  count[i+j*Nre] = testpoint(c);
}

int main(int argc, char **argv){

  // to create a 4096x4096 pixel image [ last argument is placeholder for number of threads ] 
  // usage: ./mandelbrot 4096 4096 1  
  

  int Nre = atoi(argv[1]);
  int Nim = atoi(argv[2]);
  int Nthreads = atoi(argv[3]);

  // Q2b: set the number of threads per block and the number of blocks here:

  // storage for the iteration counts
  float *h_count = (float*) malloc(Nre*Nim*sizeof(float));

  float *d_count;
  cudaMalloc(&d_count,Nre*Nim*sizeof(float));

  // Parameters for a bounding box for "c" that generates an interesting image
  const float centRe = -.759856, centIm= .125547;
  const float diam  = 0.151579;

  complex_t cmin; 
  complex_t cmax;

  cmin.r = centRe - 0.5*diam;
  cmax.r = centRe + 0.5*diam;
  cmin.i = centIm - 0.5*diam;
  cmax.i = centIm + 0.5*diam;

  //dimensions of thread blocks and grid
  dim3 B(Nthreads,Nthreads,1);
  dim3 G((Nre+Nthreads-1)/Nthreads,(Nim+Nthreads-1)/Nthreads,1);


  clock_t start = clock(); //start time in CPU cycles

  // compute mandelbrot set
  mandelbrot<<<G,B>>>(Nre, Nim, cmin, cmax, d_count); 
  cudaDeviceSynchronize(); 
  
  clock_t end = clock(); //start time in CPU cycles
  
  cudaMemcpy(d_count,h_count,Nre*Nim*sizeof(float),cudaMemcpyDeviceToHost);

  // print elapsed time
  printf("elapsed = %f\n", ((double)(end-start))/CLOCKS_PER_SEC);

  // output mandelbrot to png format image
  FILE *fp = fopen("mandelbrot.png", "w");

  printf("Printing mandelbrot.png...");
  write_hot_png(fp, Nre, Nim, h_count, 0, 80);
  printf("done.\n");

  free(h_count);
  cudaFree(h_count);

  exit(0);
  return 0;
}  
