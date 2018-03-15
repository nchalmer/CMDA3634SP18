/* 

To compile:

   gcc -O3 -o mandelbrot mandelbrot.c png_util.c -I. -lpng -lm -fopenmp

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

// Q2a: add include for OpenMP header file here:


#define MXITER 1000

typedef struct {
  
  double r;
  double i;
  
}complex_t;

// return iterations before z leaves mandelbrot set for given c
int testpoint(complex_t c){
  
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
void  mandelbrot(int Nre, int Nim, complex_t cmin, complex_t cmax, float *count){ 
  int n,m;

  complex_t c;

  double dr = (cmax.r-cmin.r)/(Nre-1);
  double di = (cmax.i-cmin.i)/(Nim-1);;

  // Q2c: add a compiler directive to split the outer for loop amongst threads here
  for(n=0;n<Nim;++n){
    for(m=0;m<Nre;++m){
      c.r = cmin.r + dr*m;
      c.i = cmin.i + di*n;
      
      count[m+n*Nre] = testpoint(c);
      
    }
  }

}

int main(int argc, char **argv){

  // to create a 4096x4096 pixel image [ last argument is placeholder for number of threads ] 
  // usage: ./mandelbrot 4096 4096 1  
  

  int Nre = atoi(argv[1]);
  int Nim = atoi(argv[2]);
  int Nthreads = atoi(argv[3]);

  // Q2b: set the number of OpenMP threads to be Nthreads here:

  // storage for the iteration counts
  float *count = (float*) malloc(Nre*Nim*sizeof(float));

  // Parameters for a bounding box for "c" that generates an interesting image
  const float centRe = -.759856, centIm= .125547;
  const float diam  = 0.151579;

  complex_t cmin; 
  complex_t cmax;

  cmin.r = centRe - 0.5*diam;
  cmax.r = centRe + 0.5*diam;
  cmin.i = centIm - 0.5*diam;
  cmax.i = centIm + 0.5*diam;

  // Q2d: complete this to read time before calling mandelbrot with OpenMP API wall clock time
  double start;

  // compute mandelbrot set
  mandelbrot(Nre, Nim, cmin, cmax, count); 
  
  // Q2d: complete this to read time after calling mandelbrot using OpenMP wall clock time
  double end;
  
  // print elapsed time
  printf("elapsed = %g\n", end-start);

  // output mandelbrot to png format image
  FILE *fp = fopen("mandelbrot.png", "w");

  write_hot_png(fp, Nre, Nim, count, 0, 80);

  exit(0);
  return 0;
}  
