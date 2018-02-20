#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "mpi.h"

int main (int argc, char **argv) {

  MPI_Init(&argc,&argv);

  int rank,size;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);

  int N;

  if (rank==0) N=199;

  /* MPI Broadcast */
  //this is the actual MPI broadcast function
  MPI_Bcast(&N,   //pointer to data
            1,    //count (number of entries)
            MPI_INT,  //data type
            0,        //root process (the process that broadcasts)
            MPI_COMM_WORLD);

  printf("Rank %d recieved the value N = %d\n",rank,N);

  if (rank==size-1) N=10;

  /* MPI Broadcast */
  MPI_Bcast(&N,   //pointer to data
            1,    //count (number of entries)
            MPI_INT,  //data type
            size-1,        //root process (the process that broadcasts)
            MPI_COMM_WORLD);
  printf("Rank %d recieved the value N = %d\n",rank,N);

  /* MPI Barrier */
  MPI_Barrier(MPI_COMM_WORLD);

  //setup a test for the reduction
  float val = 1.0;
  float sum;

  /* MPI Reduction */
  //MPI's reduce function
  MPI_Reduce(&val,    //send buffer
              &sum,   //receive buffer
              1,      //count (number of entries)
              MPI_FLOAT, //data type
              MPI_SUM,  //operation - other are MPI_MIN, MPI_MAX, MPI_PROD etc
              0,       //root process (the one that has the final answer)
              MPI_COMM_WORLD);

  for (int r=0;r<size;r++) {
    if (r==rank) {
      printf("Rank %d has value %f after the reduction.\n", rank, sum);
    }
    MPI_Barrier(MPI_COMM_WORLD); 
  }

  /* MPI Reduction + Bcast */
  //this performs the reduction so all ranks have the final value
  MPI_Allreduce(&val,    //send buffer
              &sum,   //receive buffer
              1,      //count (number of entries)
              MPI_FLOAT, //data type
              MPI_SUM,  //operation - other are MPI_MIN, MPI_MAX, MPI_PROD etc
              MPI_COMM_WORLD);

  for (int r=0;r<size;r++) {
    if (r==rank) {
      printf("Rank %d has value %f after the reduction.\n", rank, sum);
    }
    MPI_Barrier(MPI_COMM_WORLD); 
  }

  val = (float) rank;
  float *gatheredVal;

  //only rank 0 needs the storage
  if (rank==0) gatheredVal = (float *) malloc(size*sizeof(float));

  /* MPI Gather */
  //collects all data across all ranks to the root process
  MPI_Gather(&val,    //send buffer
              1,      //send count
              MPI_FLOAT,  //send type
              gatheredVal, //recv buffer
              1,            //recv count
              MPI_FLOAT,    //recv type
              0,            //root process
              MPI_COMM_WORLD);

  if (rank==0) {
    for (int r=0;r<size;r++) {
      printf("gatheredVal[%d] = %f \n", r, gatheredVal[r]);
      gatheredVal[r] *= 2;
    }
  }

  /* MPI Scatter */
  //recverse of a gather. Spreads data from the root process to all ranks
  MPI_Scatter(gatheredVal,    //send buffer
              1,      //send count
              MPI_FLOAT,  //send type
              &val, //recv buffer
              1,            //recv count
              MPI_FLOAT,    //recv type
              0,            //root process
              MPI_COMM_WORLD);

  printf("Rank %d has value %f after the scatter. \n", rank, val);

  MPI_Finalize();
  return 0;
}
