#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "omp.h"

int main(int argc, char **argv) {

  //seed random number generator
  // Q2b: get the number of threads to run with from agrv and 
  // add OpenMP API code to set number of threads here
  int Nthreads = atoi(argv[1]);
  
  omp_set_num_threads(Nthreads);

  struct drand48_data *drandData; 
  drandData = (struct drand48_data*) malloc(Nthreads*sizeof(struct drand48_data));

  // Q2c: add an OpenMP parallel region here, wherein each thread initializes 
  //      one entry in drandData using srand48_r and seed based on thread number
  #pragma omp parallel
  {
    int rank = omp_get_thread_num();
    long int seed = rank;
    srand48_r(seed, drandData+rank);
  }

  long long int Ntrials = 10000000;


  //need running tallies
  long long int Ntotal=0;
  long long int Ncircle=0;

  double startTime = omp_get_wtime();

  #pragma omp parallel for reduction(+:Ncircle) reduction(+:Ntotal)
  for (long long int n=0; n<Ntrials; n++) {
    int rank = omp_get_thread_num();

    double rand1;
    double rand2;

    //gererate two random numbers (use the thread id to offset drandData)
    drand48_r(drandData+rank, &rand1);
    drand48_r(drandData+rank, &rand2);
    
    double x = -1 + 2*rand1; //shift to [-1,1]
    double y = -1 + 2*rand2;

    //check if its in the circle
    if (sqrt(x*x+y*y)<=1) Ncircle++;
    Ntotal++;

    // if (n%100 ==0) {
    //   double pi = 4.0*Ncircle/ (double) (n);
    //   printf("Our estimate of pi is %g \n", pi);
    // }
  }

  double endTime = omp_get_wtime();

  double pi = 4.0*Ncircle/ (double) (Ntotal);
  printf("Our final estimate of pi is %g \n", pi);
  printf("Runtime was %f seconds\n", endTime-startTime);

  free(drandData);
  
  return 0;
}
