#include <stdio.h> 
#include <stdlib.h>
#include <string.h>

void myPrintf(char*); //function stub



void main() {
 
  char string[20];

  strcpy(string,"Hello World!");
 
  //call my own print function
  myPrintf(string);

}



void myPrintf(char *string) { //notice that strings are just arrays of characters

  while (*string != '\0') { //'\0' is the string's termination character

    char c = *string; //get the character at the pointer's address

    int i = (int) c;  // cast it to and integer. 
    //This interprets the 1 byte binary values of the char as an integer padded with zeros. 

    printf("%c, this char is actually a int:%d  \n", c,i); //print both the char and int to compare
    string = string +1; //shift the string's pointer address to the next character
  } 
  
}

