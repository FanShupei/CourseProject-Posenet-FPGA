CXXFLAGS = -g -I ~/Projects/ComputeLibrary -I ~/Projects/ComputeLibrary/include
LDFLAGS = -g -L ~/Projects/ComputeLibrary/build/build/x86_64-18-08 \
    -Wl,-rpath=/home/fsp/Projects/ComputeLibrary/build/build/x86_64-18-08
LDLIBS = \
    -larm_compute_core -larm_compute \
    -lopencv_highgui -lopencv_imgproc -lopencv_core

demo: v4l2.o main.o
	$(CXX) -o demo $(LDFLAGS) $(LDLIBS) v4l2.o main.o

main.o: main.cpp v4l2.h
	$(CXX) -c $(CXXFLAGS) main.cpp

v4l2.o: v4l2.cpp v4l2.h
	$(CXX) -c $(CXXFLAGS) v4l2.cpp

run: demo
	./demo
	
clean:
	rm demo
	rm *.o