#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int gcd(int a, int b) {

  if (a==0) return b;
  if (b==0) return a;

  if (b>=a) return gcd(b%a,a);
  else      return gcd(a%b,b);
}

int isCoprime(int a, int b) {

  return (gcd(a,b)==1);

}

int main(int argc, char** argv) {

  int a, b;

  printf("Enter the first number: ");
  scanf("%d",&a);

  printf("Enter the second number: ");
  scanf("%d",&b);

  if (isCoprime(a,b)) printf("%d and %d are coprime\n", a, b);
  else                printf("%d and %d are not coprime\n", a, b);

  return 0;
}