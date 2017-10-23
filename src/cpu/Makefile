CXX=g++
CXFLAGS=-Wall -Wextra -Werror
SRC=main.cpp non_local_means_cpu.cpp
OBJ=$(SRC:.c=.o)
all:exec

exec: $(OBJ)
	$(CXX) $(SRC) -o bin 
