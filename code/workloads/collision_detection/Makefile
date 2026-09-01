GENCODE_SM10 ?= -gencode=arch=compute_10,code=\"sm_10,compute_10\"
GENCODE_SM13 ?= -gencode=arch=compute_13,code=\"sm_13,compute_13\"
GENCODE_SM20 ?= -gencode=arch=compute_20,code=\"sm_20,compute_20\"
GENCODE_SM30 ?= -gencode=arch=compute_30,code=\"sm_30,compute_30\"
GENCODE_SM35 ?= -gencode=arch=compute_35,code=\"sm_35,compute_35\"
GENCODE_SM50 ?= -gencode=arch=compute_50,code=\"sm_50,compute_50\"
GENCODE_SM60 ?= -gencode=arch=compute_60,code=\"sm_60,compute_60\"
GENCODE_SM62 ?= -gencode=arch=compute_62,code=\"sm_62,compute_62\"
GENCODE_SM70 ?= -gencode=arch=compute_70,code=\"sm_70,compute_70\"
GENCODE_SM75 ?= -gencode=arch=compute_75,code=\"sm_75,compute_75\"


CPPSRC = src/main.cpp
CUSRC = src/gpu.cu
EXECUTABLE = main
RTEXECUTABLE = rtmain
OPTS = -O3 -g
CUOPTS =  $(GENCODE_ARCH) $(GENCODE_SM50) $(GENCODE_SM60) $(GENCODE_SM62)
INC = -I$(CUDA_INSTALL_PATH)/include -I./include/
LIBS = -lcudart --cudart shared

OBJS = $(notdir $(CPPSRC:.cpp=.o))

all: clean $(OBJS)
	nvcc $(NVOPTS) $(OPTS) $(CUOPTS) $(CUSRC) *.o $(INC) -o $(EXECUTABLE) $(LIBS)

rt: cleanrt $(OBJS)
	nvcc $(NVOPTS) $(OPTS) $(CUOPTS) $(CUSRC) *.o $(INC) -o $(RTEXECUTABLE) $(LIBS) -DRT_UNIT_ACCEL

$(OBJS):
	/usr/bin/g++ -c $(CPPSRC) $(OPTS) $(INC)

clean:
	rm -f *.o *.cu.cpp $(EXECUTABLE)

cleanrt:
	rm -f *.o *.cu.cpp $(RTEXECUTABLE)

cleancucpp:
	rm -f *.cu.cpp
