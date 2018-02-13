#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int gcd(int a, int b) {

  if (a==0) return b;
  if (b==0) return a;

  if (b>=a) return gcd(b%a,a);
  else      return gcd(a%b,b);
}

int lcm(int a, int b) {

  return a*b/gcd(a,b);

}

int main(int argc, char** argv) {

  int a, b;

  printf("Enter the first number: ");
  scanf("%d",&a);

  printf("Enter the second number: ");
  scanf("%d",&b);

  printf("The least common multiple of %d and %d is %d\n", a, b, lcm(a,b));

  return 0;
}