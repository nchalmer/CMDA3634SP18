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

int findGenerator(int p) {

  for (int g=2;g<p;g++) {
    int a = g;
    int i;
    for (i=2;i<p-1;i++) {
      a = (a*g)%p;
      if (a==1) { //leave the loop when we hit 1 
        break;
      }
    }
    if (i==p-1) { //if we got through the whole loop, we found a generator
      return g;
    }
  }
}

int main(int argc, char** argv) {

  int p;

  printf("Enter a prime number: ");
  scanf("%d",&p);

  if (!isPrime(p)) {
    printf("%d is not prime\n", p);
    return 1;
  }

  int g = findGenerator(p);
  printf("%d is a generator of Z_%d\n", g, p);

  return 0;
}