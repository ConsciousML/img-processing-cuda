CXX=g++
CXXFLAGS=-std=c++11 -Werror
LIBS=`pkg-config opencv --cflags --libs`
NLMSRC=bench_non_local_means_cpu.cpp non_local_means_cpu.cpp
NLMOBJ= $(NLMSRC:.c=.o)
KNNSRC=bench_knn.cpp knn.cpp
KNNOBJ= $(KNNSRC:.c=.o)
EXECS= nlm knn
nlm: $(NLMOBJ)
	$(CXX) -o $@ $^ $(LIBS) $(CXXFLAGS)
knn: $(KNNOBJ)
	$(CXX) -o $@ $^ $(LIBS) $(CXXFLAGS)
clean:
	${RM} $(EXECS)
