//compute a*b mod p safely
unsigned int modprod(unsigned int a, unsigned int b, unsigned int p);

//compute a^b mod p safely
unsigned int modExp(unsigned int a, unsigned int b, unsigned int p);

//returns either 0 or 1 randomly
unsigned int randomBit();

//returns a random integer which is between 2^{n-1} and 2^{n}
unsigned int randXbitInt(unsigned int n);

//tests for primality and return 1 if N is probably prime and 0 if N is composite
unsigned int isProbablyPrime(unsigned int N);

//Finds a generator of Z_p using the assumption that p=2*q+1
unsigned int findGenerator(unsigned int p);
