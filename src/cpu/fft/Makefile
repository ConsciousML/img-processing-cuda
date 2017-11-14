CXX=g++
INCDIR =../utils
CXXFLAGS=-Wall -Wextra -Werror -I$(INCDIR) -std=gnu++11
VPATH=$(INCDIR)
#VPATH =
SRCUTILS=$(INCDIR)/utils.cpp
SRC=test_opencv.cpp $(SRCUTILS)
CFLAGS=pkg-config --cflags opencv
LIBS=`pkg-config opencv --cflags --libs`
OBJ=$(SRC:.c=.o)
all:exec
exec: $(OBJ) 
	$(CXX) $(SRC) -o bin  $(LIBS) $(CXXFLAGS)

clean:
	rm *.o
