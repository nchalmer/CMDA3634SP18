#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int isPrime(int p) {

  int sqrtp = (int) sqrt(p);

  for (int n=2;n<=sqrtp;n++) {
    if (p%n == 0) return 0;
  }
  return 1;
}

int main(int argc, char** argv) {

  int p;

  printf("Enter a number: ");
  scanf("%d",&p);


  if (isPrime(p)) printf("%d is prime\n", p);
  else            printf("%d is not prime\n", p);

  return 0;
}