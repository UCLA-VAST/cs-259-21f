SRCS = cnn.h cnn.cpp opencl.cpp main.cpp

DSA := xilinx_aws-vu9p-f1-04261818_dynamic_5_0
DEVICE := $(AWS_PLATFORM)
VP := $(XILINX_VITIS)/bin/v++
CLFLAGS = -t $(TARGET) --platform $(DEVICE) --save-temps

export OPENCL_PLATFORM = Xilinx
export OPENCL_DEVICE = $(DSA)
export OPENCL_BINARY = cnn.$(TARGET).$(DSA).xclbin

swsim: emconfig
	XCL_EMULATION_MODE=sw_emu $(MAKE) TARGET=sw_emu dontrunme

hwsim: emconfig
	XCL_EMULATION_MODE=hw_emu $(MAKE) TARGET=hw_emu dontrunme

hls:
	$(MAKE) TARGET=hw cnn.hw.$(DSA).xo

# Building kernel
cnn.$(TARGET).$(DSA).xo: ./xilinx.cpp
	$(VP) $(CLFLAGS) -c -k CnnKernel -I'$(<D)' -o'$@' '$<'

cnn.$(TARGET).$(DSA).xclbin: cnn.$(TARGET).$(DSA).xo
	$(VP) $(CLFLAGS) -l $(LDCLFLAGS) --connectivity.nk CnnKernel:1 -o'$@' $(+)

emconfig:emconfig.json

emconfig.json:
	emconfigutil --platform $(DEVICE) --od . 


dontrunme: cnn $(OPENCL_SOURCE) cnn.$(TARGET).$(DSA).xclbin
	. ./params.sh; ./$<

cnn: $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cpp %.a %.o, $^) $(LDFLAGS)

CXXFLAGS = -std=c++11 -O3 -fopenmp -fno-loop-interchange

LDFLAGS += -lOpenCL
