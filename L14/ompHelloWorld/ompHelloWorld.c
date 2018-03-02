#include <stdio.h> 
#include <stdlib.h>
#include <math.h>

#include <omp.h>

int main (int argc, char **argv) {

  //number of parallel threads that OpenMP should use
  int NumThreads = 4;
  
  //tell OpenMP to use NumThreads threads
  omp_set_num_threads(NumThreads);

  #pragma omp parallel
  {
    int rank = omp_get_thread_num();  //thread's rank
    int size = omp_get_num_threads(); //total number of threads
   
    printf("Hello World from thread %d of %d \n", rank, size);
  }

  return 0;
}

