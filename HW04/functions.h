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

//Sets up an ElGamal cryptographic system
void setupElGamal(unsigned int n, unsigned int *p, unsigned int *g, 
                                  unsigned int *h, unsigned int *x);

//encrypt a number *m using ElGamal and return the 
//  coefficient *a used in the encryption.
void ElGamalEncrypt(unsigned int *m, unsigned int *a, unsigned int Nints,
                    unsigned int p, unsigned int g, unsigned int h);

//decryp a number *m using ElGamal using the coefficent
//  *a and the secret key x.
void ElGamalDecrypt(unsigned int *m, unsigned int *a, unsigned int Nints,
                    unsigned int p, unsigned int x);


//Pad the end of string so its length is divisible by Nchars
// Assume there is enough allocated storage for the padded string 
void padString(unsigned char* string, unsigned int charsPerInt);

void convertStringToZ(unsigned char *string, unsigned int Nchars,
                      unsigned int  *Z,      unsigned int Nints);

void convertZToString(unsigned int  *Z,      unsigned int Nints,
                      unsigned char *string, unsigned int Nchars);
