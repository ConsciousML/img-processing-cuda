CXX=g++
INCDIR =../
CXXFLAGS= -I$(INCDIR) -std=c++11
LIBS=`pkg-config opencv --cflags --libs`
NLMSRC=bench_non_local_means_cpu.cpp ../non_local_means_cpu.cpp
NLMOBJ= $(NLMSRC:.c=.o)
KNNSRC=bench_knn.cpp ../knn.cpp
KNNOBJ= $(KNNSRC:.c=.o)
EDGSRC=bench_edge_detect.cpp ../edge_detect.cpp
EDGOBJ= $(EDGSRC:.c=.o)
EXECS= nlm knn edge
nlm: $(NLMOBJ)
	$(CXX) -o $@ $^ $(LIBS) $(CXXFLAGS)
knn: $(KNNOBJ)
	$(CXX) -o $@ $^ $(LIBS) $(CXXFLAGS)
edge: $(EDGOBJ)
	$(CXX) -o $@ $^ $(LIBS) $(CXXFLAGS)
clean:
	${RM} $(EXECS)
