#include "mympi.h"

void treeMPI_Bcast(int *N) {
  //get the rank of this MPI process and the total number of ranks 
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);

  //start with only one rank has the message
  int Nactive = 1; 

  //while not all ranks have the messsage 
  while (Nactive < size) {
  
    //ranks < Nactive send the value
    if (rank < Nactive) {
      //MPI_Send to rank + Nactive
      int destRank = rank + Nactive;

      if (destRank < size) {
        int tag = Nactive;
        
        MPI_Send(N,
                  1,
                  MPI_INT,
                  destRank,
                  tag,
                  MPI_COMM_WORLD);
      }
    } 
    if ((rank >= Nactive)&&(rank < 2*Nactive))  {
      //MPI_recv from rank - Nactive
      int sourceRank = rank - Nactive;
      int tag = Nactive;
      MPI_Status status;

      MPI_Recv(N,
                1,
                MPI_INT,
                sourceRank,
                tag,
                MPI_COMM_WORLD,
                &status);
  
    }
    Nactive *= 2;  //double the number of ranks that are active
  } 
}

