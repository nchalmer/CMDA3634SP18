#include <stdio.h>
#include <stdlib.h>

void main() {
  
  int a,b,c; //allocates an integer called 'a'
  int* pt_a, *pt_b; //alloccates a pointer, called pt_a, to an interger


  a=13;
  b=4;
  pt_a = &a; //stores not the value of a, but its location 
  pt_b = &b; //stores not the value of b, but its location

  c = *(pt_a+1); 

  printf("The size of an int is %ld\n",sizeof(int));
  printf("a is located at %p\n",pt_a);
  printf("b is located at %p\n",pt_b);
  printf("a = %d \n",a);
  printf("b = %d \n",b);
  printf("c = %d \n",c);
  printf("a+b = %d \n",a+b);
  printf("a-b = %d \n",a-b);
  printf("a*b = %d \n",a*b);
  printf("a/b = %d \n",a/b);
  printf("a mod b = %d \n",a%b);

  int *array;

  array = (int *) malloc(10*sizeof(int));

  printf("a[0] = %d\n", array[0]); 
  printf("a[1] = %d\n", array[1]); 
  printf("a[2] = %d\n", array[2]); 
  printf("a[3] = %d\n", array[3]); 
  printf("a[4] = %d\n", array[4]); 
  printf("a[5] = %d\n", array[5]); 
  printf("a[6] = %d\n", array[6]); 
  printf("a[7] = %d\n", array[7]); 
  printf("a[8] = %d\n", array[8]); 
  printf("a[9] = %d\n", array[9]); 
}
