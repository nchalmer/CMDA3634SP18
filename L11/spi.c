#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char **argv) {

  //need running tallies
  long long int Ntotal;
  long long int Ncircle;

  //seed random number generator
  double seed = 1.0;
  srand48(seed);

  for (long long int n=0; n<1000000000;n++) {
    //gererate two random numbers
    double rand1 = drand48(); //drand48 returns a number between 0 and 1
    double rand2 = drand48();
    
    double x = -1 + 2*rand1; //shift to [-1,1]
    double y = -1 + 2*rand2;

    //check if its in the circle
    if (sqrt(x*x+y*y)<=1) Ncircle++;
    Ntotal++;
  }

  double pi = 4.0*Ncircle/ (double) Ntotal;

  printf("Our estimate of pi is %f \n", pi);

  return 0;
}
