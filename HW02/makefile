#can change these if files are located elsewhere
SOURCEDIR = ./.
INCLUDEDIR = ./.
OBJDIR = ./.


SOURCE = $(wildcard $(SOURCEDIR)/*c)
INCLUDES = $(wildcard $(INCLUDEDIR)/*h)
OBJECTS = $(subst $(SOURCEDIR)/,$(OBJDIR)/,$(SOURCE:.c=.o))

DEPS = $(INCLUDES) 
FLAGS = -I$(INCLUDEDIR) -O3 -g
LIBS  = -lm

CC =  gcc

$(OBJDIR)/%.o:$(SOURCEDIR)/%.c $(DEPS)
	$(CC) $(FLAGS) -o $@ $(LIBS) -c $<

all: main

main: $(OBJECTS) $(DEPS)
	$(CC)  $(OBJECTS) $(FLAGS) -o main $(LIBS)

clean:
	rm -f main
	rm -f $(OBJDIR)/*.o
