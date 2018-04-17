#can change these if files are located elsewhere
SOURCEDIR = ./.
INCLUDEDIR = ./.
OBJDIR = ./.


INCLUDES = $(wildcard $(INCLUDEDIR)/*h)
SOURCE = functions.c
OBJECTS = $(subst $(SOURCEDIR)/,$(OBJDIR)/,$(SOURCE:.c=.o))

DEPS = $(INCLUDES) 
FLAGS = -I$(INCLUDEDIR) -g -O3 
NVFLAGS = -arch=sm_60
LIBS  = -lm

CC =  gcc
NVCC =  nvcc

$(OBJDIR)/%.o:$(SOURCEDIR)/%.cu $(DEPS)
	$(CC) $(FLAGS) -o $@ $(LIBS) -c $<

all: setup encrypt decrypt cudaDecrypt

setup:  setup.o functions.o $(DEPS)
	$(CC) setup.o functions.o $(FLAGS) -o setup $(LIBS)

encrypt: encrypt.o functions.o $(DEPS)
	$(CC)  encrypt.o functions.o $(FLAGS) -o encrypt $(LIBS)

decrypt: decrypt.o functions.o $(DEPS)
	$(CC)  decrypt.o functions.o $(FLAGS) -o decrypt $(LIBS)

cudaDecrypt: cudaDecrypt.cu functions.o $(DEPS)
	$(NVCC)  cudaDecrypt.cu $(FLAGS) $(NVFLAGS) -o cudaDecrypt $(LIBS)

clean:
	rm -f setup encrypt decrypt cudaDecrypt
	rm -f $(OBJDIR)/*.o
