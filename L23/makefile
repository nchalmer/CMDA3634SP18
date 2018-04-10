#can change these if files are located elsewhere
SOURCEDIR = ./.
INCLUDEDIR = ./.
OBJDIR = ./.


SOURCE = $(wildcard $(SOURCEDIR)/*cu)
INCLUDES = $(wildcard $(INCLUDEDIR)/*h)
OBJECTS = $(subst $(SOURCEDIR)/,$(OBJDIR)/,$(SOURCE:.cu=.o))

DEPS = $(INCLUDES) 
FLAGS = -I$(INCLUDEDIR) -O3 -g -arch=sm_60
LIBS  = -lm

CC =  nvcc 

$(OBJDIR)/%.o:$(SOURCEDIR)/%.cu $(DEPS)
	$(CC) $(FLAGS) -o $@ $(LIBS) -c $<

all: main

main: $(OBJECTS) $(DEPS)
	$(CC)  $(OBJECTS) $(FLAGS) -o main $(LIBS)

clean:
	rm -f main
	rm -f $(OBJDIR)/*.o
