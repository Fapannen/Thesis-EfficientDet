INC=-I/home/fapannen/Desktop/git/tensorflow-imx \
	-I/home/fapannen/Desktop/git/tensorflow-imx/tensorflow/lite/tools/make/downloads/flatbuffers/include

LIBS=-ltensorflow-lite \
	-lstdc++ \
	-lpthread \
	-lrt \
	-ldl \
	-lm \
	-lz

BIN=measure

EXT=../../tensorflow-imx/tensorflow/lite/nnapi/nnapi_implementation.cc

LDOPTS:=-L.


measure: $(BIN).cpp
	$(CXX) -O2 $(INC) $(EXT) $(BIN).cpp $(LDOPTS) $(LIBS) -o $(BIN)
