#include<stdio.h>
#include<stdlib.h>
#include <math.h>

//define a new data structure called 'point'
typedef struct {

  float x;
  float y;
  float z;

} point;


//we can pass this structure to functions and access it's sub-variables
void pointPrintPoint(point p) {

  printf("Point has coordinates (%f,%f,%f) \n", p.x,p.y,p.z);

}

//if we want to alter the structure inside a function, we need to pass its pointer
void pointSetZero(point *p) {

  //(*p).x = 0.;
  //(*p).y = 0.;
  //(*p).z = 0.;
  
  //also valid
  //point pp = *p;
  //pp.x = 0.;
  //pp.y = 0.;
  //pp.z = 0.;
  
  //and so is this. -> is a kind of shift+dereferance operator
  p->x = 0.;
  p->y = 0.;
  p->z = 0.;
} 

float pointDistanceToOrigin(point p) {

  float dist = sqrt(p.x*p.x+p.y*p.y+p.z*p.z);

  return dist;

}

void main() {
 //we can declare this new structure like any other variable
  point p;

  //access variables in stucture with '.'
  p.x = 1.0;
  p.y = 2.0;
  p.z = 3.0;
  
  float dist;

  //we can pass it to a function and return a value
  dist = pointDistanceToOrigin(p);

  printf("dist = %f\n",dist);

  //we pass its pointer so we can change it
  pointSetZero(&p);

  pointPrintPoint(p);
}

