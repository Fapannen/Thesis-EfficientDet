INC=-I/home/fapannen/Desktop/git/tensorflow-imx \
	-I/home/fapannen/Desktop/git/tensorflow-imx/tensorflow/lite/tools/make/downloads/flatbuffers/include

LIBS=-ltensorflow-lite \
	-lstdc++ \
	-lpthread \
	-lrt \
	-ldl \
	-lm \
	-lz \
	-lopencv_dnn \
	-lopencv_ml \
	-lopencv_shape \
	-lopencv_stitching \
	-lopencv_superres \
	-lopencv_videostab \
	-lopencv_calib3d \
	-lopencv_features2d \
	-lopencv_highgui \
	-lopencv_videoio \
	-lopencv_photo \
	-lopencv_imgcodecs \
	-lopencv_video \
	-lopencv_objdetect \
	-lopencv_imgproc \
	-lopencv_flann \
	-lopencv_core

BIN=measure

EXT=../../../tensorflow-imx/tensorflow/lite/nnapi/nnapi_implementation.cc

LDOPTS:=-L./libs

UTILS=utils

measure: $(BIN).cpp $(UTILS).cpp $(UTILS).hpp
	$(CXX) -O2 $(INC) $(EXT) $(UTILS).cpp $(BIN).cpp $(LDOPTS) $(LIBS) -o $(BIN)
