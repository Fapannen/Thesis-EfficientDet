INC=-I/mnt/c/Users/Owner/Desktop/git/tensorflow \
	-I/mnt/c/Users/Owner/Desktop/git/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include \
	-I/usr/local/include/opencv4
	
LIBS=-ltensorflow-lite \
	-lstdc++ \
	-lstdc++fs \
	-lpthread \
	-lrt \
	-ldl \
	-lm \
	-lz \
	-lopencv_stitching \
	-lopencv_calib3d \
	-lopencv_features2d \
	-lopencv_highgui \
	-lopencv_photo \
	-lopencv_imgcodecs \
	-lopencv_imgproc \
	-lopencv_flann \
	-lopencv_core

BIN=measure

EXT=../../../tensorflow/tensorflow/lite/nnapi/nnapi_implementation.cc

LDOPTS:=-L./libs

UTILS=utils

efficientdet: $(BIN)_efficientdet.cpp $(UTILS).cpp $(UTILS).hpp
	$(CXX) -std=c++17 -O2 $(INC) $(EXT) $(UTILS).cpp $(BIN)_efficientdet.cpp $(LDOPTS) $(LIBS) -o $(BIN)_efficientdet

mobilenet: $(BIN)_mobilenet.cpp $(UTILS).cpp $(UTILS).hpp
	$(CXX) -std=c++17 -O2 $(INC) $(EXT) $(UTILS).cpp $(BIN)_mobilenet.cpp $(LDOPTS) $(LIBS) -o $(BIN)_mobilenet

