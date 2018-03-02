#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "mpi.h"

int main(int argc, char **argv) {

  MPI_Init(&argc,&argv);
  
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);

  //need running tallies
  long long int Ntotal=0;
  long long int Ncircle=0;

  //seed random number generator
  double seed = rank;
  srand48(seed);

  long long int Ntrials = 10000000;

  MPI_Barrier(MPI_COMM_WORLD);
  double startTime = MPI_Wtime();

  for (long long int n=0; n<Ntrials; n++) {
    //gererate two random numbers
    double rand1 = drand48(); //drand48 returns a number between 0 and 1
    double rand2 = drand48();
    
    double x = -1 + 2*rand1; //shift to [-1,1]
    double y = -1 + 2*rand2;

    //check if its in the circle
    if (sqrt(x*x+y*y)<=1) Ncircle++;
    Ntotal++;
  }

  long long int globalNcircle=0;
  MPI_Allreduce(&Ncircle, &globalNcircle, 1,
                  MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);

  double pi = 4.0*globalNcircle/ (double) (Ntotal*size);
  
  MPI_Barrier(MPI_COMM_WORLD);
  double endTime = MPI_Wtime();  

  long long int totalWork = size*Ntrials;

  if (rank==0) 
    printf("Our estimate of pi is %f, the runtime was %g, our 'throughput' was %g \n", pi, endTime-startTime, totalWork/(endTime-startTime));
  
  MPI_Finalize();

  return 0;
}
