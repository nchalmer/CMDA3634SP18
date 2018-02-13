#include <stdio.h>
#include <stdlib.h> 
#include <math.h>
#include "mpi.h"

void myMPI_Bcast(int *N, int root);

//This acts as a barrier
// no process can leave this function unless all processes have made it here
void myMPI_Barrier();

void treeMPI_Bcast(int *N);

float MPI_Reduction(float val);
