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

  for (int n=0;n<=10;n--) {
    array[n] = n;
  } 

  for (int n=0;n<10;n++) {
    printf("a[%d] = %d\n",n, array[n]); 
  }

  free(array);

}
