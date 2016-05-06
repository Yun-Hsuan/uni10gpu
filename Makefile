GPU_LIB := libuni10
UNI10_SRC_ROOT :=/home/Yun-Hsuan/GitRepo/tensorlib/uni10/src/uni10
SRC := .

OPENBLASDIR  ?= /usr/local/openblas
MAGMA_TEST :=/home/Yun-Hsuan/INSTALL/MAGMA/magma-2.0.1/testing/libtest.a

CC            =	icpc
FORT          = gfortran
LD            = icpc
NVCC          = nvcc
CFLAGS        = -Wall -std=c++11 -O3 -m64
LDFLAGS       = -Wall -std=c++11 -O3 -m64
NVCCOPT       = -ccbin icpc -Xcompiler "-Wall -Wno-long-long -ansi -pedantic -ansi-alias -parallel -fopenmp -openmp-link=static -static-intel -wd10237" -O3 -Xcompiler "-O3" -m64 -arch=sm_30

MAGMA_CFLAGS   := -DADD_ -I$(MAGMADIR)/include -I$(CUDADIR)/include
MAGMA_F90FLAGS := -I$(MAGMADIR)/include -Dmagma_devptr_t="integer(kind=8)"

MAGMA_LIBS   := -L$(MAGMADIR)/lib -L$(CUDADIR)/lib64 -L$(OPENBLASDIR)/lib \
                -lmagma -lcublas -lcudart -lopenblas


MKL_LIB1 := /opt/intel/composer_xe_2015.3.187/mkl/lib/intel64/libmkl_intel_lp64.a
MKL_LIB2 := /opt/intel/composer_xe_2015.3.187/mkl/lib/intel64/libmkl_intel_thread.a
MKL_LIB3 := /opt/intel/composer_xe_2015.3.187/mkl/lib/intel64/libmkl_core.a

INC := $(UNI10_ROOT)/include
LIB := ./libuni10_gpu.a

# Alternatively, using pkg-config (see README.txt):
# MAGMA_CFLAGS := $(shell pkg-config --cflags magma)
# MAGMA_LIBS   := $(shell pkg-config --libs   magma)
#

GPU_OBJ := $(GPU_LIB)/qnum.o $(GPU_LIB)/bond.o $(GPU_LIB)/block.o $(GPU_LIB)/blockreal.o $(GPU_LIB)/blockcomplex.o $(GPU_LIB)/blocktools.o $(GPU_LIB)/uni10_tools.o $(GPU_LIB)/uni10_tools_gpu.o $(GPU_LIB)/uni10_dgemm.o $(GPU_LIB)/uni10_lapack_gpu.o $(GPU_LIB)/hdf5io.o $(GPU_LIB)/matrix.o $(GPU_LIB)/matrixreal.o $(GPU_LIB)/matrixcomplex.o $(GPU_LIB)/matrixtools.o $(GPU_LIB)/UniTensor.o $(GPU_LIB)/UniTensorreal.o $(GPU_LIB)/UniTensorcomplex.o $(GPU_LIB)/UniTensortools.o $(GPU_LIB)/network.o

gpu: libuni10_gpu.a

$(GPU_LIB)/qnum.o: $(UNI10_SRC_ROOT)/datatype/lib/Qnum.cpp $(UNI10_SRC_ROOT)/datatype/Qnum.h
	$(CC) -c $(CFLAGS) -I $(SRC) $< -o $@
$(GPU_LIB)/bond.o: $(UNI10_SRC_ROOT)/data-structure/lib/Bond.cpp $(UNI10_SRC_ROOT)/data-structure/Bond.h
	$(CC) -c $(CFLAGS) -I $(SRC) $< -o $@
$(GPU_LIB)/block.o: $(UNI10_SRC_ROOT)/data-structure/lib/Block.cpp $(UNI10_SRC_ROOT)/datatype.hpp $(UNI10_SRC_ROOT)/data-structure/Block.h
	$(CC) -c $(CFLAGS) -I $(SRC) $< -o $@
$(GPU_LIB)/blockreal.o: $(UNI10_SRC_ROOT)/data-structure/lib/BlockReal.cpp $(UNI10_SRC_ROOT)/data-structure/Block.h
	$(CC) -c $(CFLAGS) -I $(SRC) $< -o $@
$(GPU_LIB)/blockcomplex.o: $(UNI10_SRC_ROOT)/data-structure/lib/BlockComplex.cpp $(UNI10_SRC_ROOT)/data-structure/Block.h
	$(CC) -c $(CFLAGS) -I $(SRC) $< -o $@
$(GPU_LIB)/blocktools.o: $(UNI10_SRC_ROOT)/data-structure/lib/BlockTools.cpp $(UNI10_SRC_ROOT)/data-structure/Block.h
	$(CC) -c $(CFLAGS) -I $(SRC) $< -o $@
$(GPU_LIB)/uni10_tools.o: $(UNI10_SRC_ROOT)/tools/lib/uni10_tools.cpp $(UNI10_SRC_ROOT)/tools/uni10_tools.h
	$(CC) -c $(CFLAGS) -I $(SRC) $< -o $@
$(GPU_LIB)/uni10_tools_cpu.o: $(UNI10_SRC_ROOT)/tools/lib/uni10_tools_cpu.cpp $(UNI10_SRC_ROOT)/tools/uni10_tools.h
	$(CC) -c $(CFLAGS) -I $(SRC) $< -o $@
$(GPU_LIB)/uni10_tools_gpu.o: $(UNI10_SRC_ROOT)/tools/lib/uni10_tools_gpu.cu $(UNI10_SRC_ROOT)/tools/uni10_tools.h
	$(NVCC) -c $(NVCCOPT) -I $(SRC) $< -o $@
$(GPU_LIB)/uni10_dgemm.o: $(UNI10_SRC_ROOT)/numeric/lapack/lib/uni10_dgemm.cu $(UNI10_SRC_ROOT)/numeric/lapack/uni10_lapack.h
	$(NVCC) -c $(NVCCOPT) -I $(SRC) $< -o $@
$(GPU_LIB)/uni10_lapack_cpu.o: $(UNI10_SRC_ROOT)/numeric/lapack/lib/uni10_lapack_cpu.cpp $(UNI10_SRC_ROOT)/numeric/lapack/uni10_lapack.h $(UNI10_SRC_ROOT)/numeric/lapack/uni10_lapack_wrapper.h
	$(CC) -c $(CFLAGS) -I $(SRC) $< -o $@
$(GPU_LIB)/uni10_lapack_gpu.o: $(UNI10_SRC_ROOT)/numeric/lapack/lib/uni10_lapack_gpu.cu $(UNI10_SRC_ROOT)/numeric/lapack/uni10_lapack.h
	$(NVCC) -c $(NVCCOPT) -I $(SRC) $< -o $@
$(GPU_LIB)/hdf5io.o: $(UNI10_SRC_ROOT)/hdf5io/lib/uni10_hdf5io.cpp $(UNI10_SRC_ROOT)/hdf5io/uni10_hdf5io.h
	$(CC) -c $(CFLAGS) -I $(SRC) $< -o $@
$(GPU_LIB)/matrix.o: $(UNI10_SRC_ROOT)/tensor-network/lib/Matrix.cpp $(UNI10_SRC_ROOT)/tensor-network/Matrix.h
	$(CC) -c $(CFLAGS) -I $(SRC) $< -o $@
$(GPU_LIB)/matrixreal.o: $(UNI10_SRC_ROOT)/tensor-network/lib/MatrixReal.cpp $(UNI10_SRC_ROOT)/tensor-network/Matrix.h
	$(CC) -c $(CFLAGS) -I $(SRC) $< -o $@
$(GPU_LIB)/matrixcomplex.o: $(UNI10_SRC_ROOT)/tensor-network/lib/MatrixComplex.cpp $(UNI10_SRC_ROOT)/tensor-network/Matrix.h
	$(CC) -c $(CFLAGS) -I $(SRC) $< -o $@
$(GPU_LIB)/matrixtools.o: $(UNI10_SRC_ROOT)/tensor-network/lib/MatrixTools.cpp $(UNI10_SRC_ROOT)/tensor-network/Matrix.h
	$(CC) -c $(CFLAGS) -I $(SRC) $< -o $@
$(GPU_LIB)/UniTensor.o: $(UNI10_SRC_ROOT)/tensor-network/lib/UniTensor.cpp $(UNI10_SRC_ROOT)/tensor-network/UniTensor.h
	$(CC) -c $(CFLAGS) -I $(SRC) $< -o $@
$(GPU_LIB)/UniTensorreal.o: $(UNI10_SRC_ROOT)/tensor-network/lib/UniTensorReal.cpp $(UNI10_SRC_ROOT)/tensor-network/UniTensor.h
	$(CC) -c $(CFLAGS) -I $(SRC) $< -o $@
$(GPU_LIB)/UniTensorcomplex.o: $(UNI10_SRC_ROOT)/tensor-network/lib/UniTensorComplex.cpp $(UNI10_SRC_ROOT)/tensor-network/UniTensor.h
	$(CC) -c $(CFLAGS) -I $(SRC) $< -o $@
$(GPU_LIB)/UniTensortools.o: $(UNI10_SRC_ROOT)/tensor-network/lib/UniTensorTools.cpp $(UNI10_SRC_ROOT)/tensor-network/UniTensor.h
	$(CC) -c $(CFLAGS) -I $(SRC) $< -o $@
$(GPU_LIB)/network.o: $(UNI10_SRC_ROOT)/tensor-network/lib/Network.cpp $(UNI10_SRC_ROOT)/tensor-network/Network.h
	$(CC) -c $(CFLAGS) -I $(SRC) $< -o $@

libuni10_gpu.a: $(GPU_OBJ)
	ar rcs $(GPU_LIB)/$@ $^

# ----------------------------------------

all: exu obj

obj: gpu

exu: transpose

clean-exu:
	-rm -f transpose
clean:
	-rm -f $(GPU_LIB)/*.o $(GPU_LIB)/*.a *.mod

.SUFFIXES:


# ----------------------------------------
# C++ example
%.o: %.cu
	$(CC) $(CFLAGS) $(MAGMA_CFLAGS) -c -o $@ $<

transpose: transpose.cu
	$(NVCC) -ftz true $(NVCCOPT) $< -o $@ $(GPU_OBJ) -lcublas -lcusolver -lcusparse -lblas -llapack -lm -lhdf5_cpp -lhdf5

