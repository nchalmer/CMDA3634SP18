#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "functions.h"

int main (int argc, char **argv) {

	//seed value for the randomizer 
  double seed;
  
  seed = clock(); //this will make your program run differently everytime
  //seed = 0; //uncomment this and you program will behave the same everytime it's run
  
  srand(seed);


  //begin by getting user's input
	unsigned int n;

  printf("Enter a number of bits: ");
  char status = scanf("%u",&n);

  //make sure the input makes sense
  if ((n<2)||(n>31)) { //Update: 31 bits works 
  	printf("Unsupported bit size.\n");
		return 0;  	
  }

  int p;

  /* Use isProbablyPrime and randomXbitInt to find a random n-bit prime number */
  do {
    p = randXbitInt(n);
  } while (!isProbablyPrime(p));

  printf("p = %u is probably prime.\n", p);

  /* Use isProbablyPrime and randomXbitInt to find a new random n-bit prime number 
     which satisfies p=2*q+1 where q is also prime */
  int q;

  do {
    p = randXbitInt(n);
    q = (p-1)/2;
  } while (!isProbablyPrime(p) || !isProbablyPrime(q));

	printf("p = %u is probably prime and equals 2*q + 1. q= %u and is also probably prime.\n", p, q);  

	/* Use the fact that p=2*q+1 to quickly find a generator */
	unsigned int g = findGenerator(p);

	printf("g = %u is a generator of Z_%u \n", g, p);  

  /* BONUS! */
  //pick a random x
  unsigned int x = randXbitInt(n)%p;

  //compute h
  unsigned int h = modExp(g,x,p);

  printf("Secret key = %u, h = g^x = %u\n", x, h);
  printf("\n");

  //now suppose we don't know x and search for it with a loop
  for (unsigned int x=1;x<p;x++) {
    if (modExp(g,x,p)==h)
      printf("Secret key found! x = %u \n", x);
  }

  return 0;
}
