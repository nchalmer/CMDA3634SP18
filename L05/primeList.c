#include<stdio.h>
#include<stdlib.h>
#include<math.h>


void main() {

  int N;

  printf("Enter an upper bound:");
  scanf("%d",&N);

  // make storage for flags
  int *isPrime = (int*) malloc(N*sizeof(int));

  //initialize, i.e. set everything 'true'
  for (int n=0;n<N;n++) isPrime[n] = 1; 

  int sqrtN = (int) sqrt(N);

  for (int i=2;i<sqrtN;i++) {
    if (isPrime[i]) { //if i is prime
      for (int j=i*i;j<N;j++) {
        isPrime[j] = 0;//set j not prime
      }
    }
  }

  // count the number of primes we found 
  int cnt =0;
  for (int n=0;n<N;n++) {
    if (isPrime[n]) {
      cnt++;
    }
  } 

  //make a list of them
  int *primes = (int*) malloc(cnt*sizeof(int));

  //loop once more and build the list
  cnt =0;
  for (int n=0;n<N;n++) {
    if (isPrime[n]) {
      primes[cnt++] = n;
    }
  } 

  //print out what find 
  for (int n=0;n<cnt;n++) printf("The %d-th prime is %d\n", n, primes[n]);
  
}
