CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra

shader: shader.cpp bvh.cpp bvh.h mesh.h common.h
	$(CXX) $(CXXFLAGS) shader.cpp bvh.cpp -o $@

render: shader
	./shader > out.ppm

clean:
	rm -f shader out.ppm

.PHONY: render clean
