#include "mympi.h"

//compute the global sum of all the values in val across all MPI ranks
float MPI_Reduction(float val) {

  float sum = val;

  float recvBuffer; //space for incoming values
  
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);

  //start with all ranks active
  int Nactive = size;

  while (Nactive > 1) {
  
    if (rank >= (Nactive+1)/2) {
      //MPI_send to rank - Nactive/2
      int destRank = rank - (Nactive+1)/2;
      int tag = Nactive;
      
      MPI_Send(&sum,
                1,
                MPI_FLOAT,
                destRank,
                tag,
                MPI_COMM_WORLD);
    }
    if (rank < Nactive/2) {
      //MPI_recv from rank + Nactive/2
      int sourceRank = rank + (Nactive+1)/2;
      int tag = Nactive;
      MPI_Status status;

      //careful for the case when Nactive is odd. Check if sourceRank is valid
      MPI_Recv(&recvBuffer,
                1,
                MPI_FLOAT,
                sourceRank,
                tag,
                MPI_COMM_WORLD,
                &status);     
      //add the recvied value to the sum
      sum += recvBuffer; 
    }
    
    //only half the ranks are active next time
    Nactive = (Nactive+1)/2;
  }
  return sum;
}
