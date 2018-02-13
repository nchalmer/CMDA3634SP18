#include<stdlib.h>
#include<stdio.h>
#include<math.h>

#include "mpi.h"


int main(int argc, char** argv) {

  //every MPI program must start with an initialize
  //always do this first
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD,  //This tells MPI to get the rank of this process globally
                &rank);          //Store the result in rank
  MPI_Comm_size(MPI_COMM_WORLD,  //This tells MPI to get the total number of processes
                &size);          //Store the result in size


  if (rank==0) {
    int N = 10;
    int destRank = 1;
    int tag = 1;
    MPI_Request reqs[2];    
    MPI_Status status[2]; 

    int *sendData = (int *) malloc(N*sizeof(int));
  
    for (int n=0;n<N;n++) sendData[n] = 2*n;

    // issue a send request
    MPI_Isend(&N,              //pointer to the data we're sending
             1,               //number of entries to send
             MPI_INT,         //data type of each entry
             destRank,        // rank of destination
             tag,             // tags the message with an identifier 
             MPI_COMM_WORLD, // flag to full MPI network
             reqs+0);

    tag = 2; //new tag
    // issue a send request
    MPI_Isend(sendData,              //pointer to the data we're sending
             N,               //number of entries to send
             MPI_INT,         //data type of each entry
             destRank,        // rank of destination
             tag,             // tags the message with an identifier 
             MPI_COMM_WORLD, // flag to full MPI network
             reqs+1);


    //wait for the request to compete before proceeding  
    MPI_Wait(reqs+0, status+0);    
    MPI_Wait(reqs+1, status+1);
     
    //free(sendData);
  } else if (rank ==1) {
    
    int N;
    int sourceRank = 0;
    int tag = 1;
    MPI_Request reqs[2];    
    MPI_Status status[2]; 

    MPI_Irecv(&N,              //pointer to where to put data 
             1,               // number of entries
             MPI_INT,         // data type
             sourceRank,      // rank of sender
             tag,             // message tag
             MPI_COMM_WORLD,
             reqs+0);
    
    //wait for the request to compete before proceeding  
    MPI_Wait(reqs+0, status+0);    

    printf("Rank %d recieved a message from rank %d: value = %d\n", rank, sourceRank, N);
  
    int *recvData = (int *) malloc(N*sizeof(int));
    
    tag =2;
    MPI_Irecv(recvData,              //pointer to where to put data 
             N,               // number of entries
             MPI_INT,         // data type
             sourceRank,      // rank of sender
             tag,             // message tag
             MPI_COMM_WORLD,
             reqs+1);

    MPI_Wait(reqs+1, status+1);    

    for (int n=0;n<N;n++) { 
      printf("Rank %d recieved a message from rank %d: data[%d] = %d\n", rank, sourceRank, n,recvData[n]);
    } 
    //free(recvData);
  }

  // all MPI programs must end with a finalize
  MPI_Finalize();
  return 0;
}
