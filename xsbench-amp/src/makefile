#===============================================================================
# User Options
#===============================================================================

COMPILER     = hcc
MODEL        = lc
#HCC_ARR_TYPE = array_view
#HCC_ARR_TYPE = array
HCC_ARR_TYPE = am_alloc
WHICH_ACC    = 0
OPTIMIZE     = no
DEBUG        = no
PROFILE      = no
MPI          = no
PAPI         = no
VEC_INFO     = no

#Does not work for many sizes on the APU, dependent on dedicated GPU memory
VERIFY       = yes

BENCHMARK    = no
BINARY_DUMP  = no
BINARY_READ  = no

#===============================================================================
# Program name & source code list
#===============================================================================

program = XSBench

source = \
Main.cpp \
io.cpp \
GridInit.cpp \
XSutils.cpp \
Materials.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := -std=gnu99

# Linker Flags
LDFLAGS = -lm

# C++AMP hcc Compiler
ifeq ($(COMPILER),hcc)

  ifeq ($(MODEL),lc)
    #dGPU
    HCC_HOME = /opt/rocm/hcc
  else
  ifeq ($(MODEL),hsail)
    #APU
    HCC_HOME = /opt/rocm/hcc-hsail
  else
    #assume OLD, pre-ROCM
    HCC_HOME = /opt/hcc
  endif
  endif

  INCLUDES = -I.

  CC = $(HCC_HOME)/bin/hcc
  LDFLAGS = $(shell $(HCC_HOME)/bin/hcc-config --install --ldflags)
  LDLIBS  = -lpthread -lm -lmpi -ldl -lc -fopenmp -lhc_am
  CFLAGS = -fopenmp -O3 -Wno-parentheses \
           $(shell $(HCC_HOME)/bin/hcc-config --install --cxxflags)

  ifeq ($(HCC_ARR_TYPE),array_view)
    CFLAGS += -DARRAY_TYPE=0
  endif
  ifeq ($(HCC_ARR_TYPE),array)
    CFLAGS += -DARRAY_TYPE=1
  endif
  ifeq ($(HCC_ARR_TYPE),am_alloc)
    CFLAGS += -DARRAY_TYPE=2
  endif

  CFLAGS += -DWHICH_ACC=$(WHICH_ACC)

endif

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS  += -g
endif

# Profiling Flags
ifeq ($(PROFILE),yes)
  CFLAGS += -pg
  LDFLAGS  += -pg
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

# Compiler Vectorization (needs -O3 flag) information
ifeq ($(VEC_INFO),yes)
  CFLAGS += -ftree-vectorizer-verbose=6
endif

# PAPI source (you may need to provide -I and -L pointing
# to PAPI depending on your installation
ifeq ($(PAPI),yes)
  source += papi.c
  CFLAGS += -DPAPI
  #CFLAGS += -I/soft/apps/packages/papi/papi-5.1.1/include
  #LDFLAGS += -L/soft/apps/packages/papi/papi-5.1.1/lib -lpapi
  LDFLAGS += -lpapi
endif

# MPI
ifeq ($(MPI),yes)
  MPI_INCLUDE = /usr/include/mpich
  MPI_LIB_DIR = /usr/lib/mpich/
  LDFLAGS += -L$(MPI_LIB_DIR)
  CFLAGS += -DDOMPI -I$(MPI_INCLUDE)
  CXXFLAGS += -DDOMPI -I$(MPI_INCLUDE)
endif

# Verification of results mode
ifeq ($(VERIFY),yes)
  CFLAGS += -DVERIFICATION
  CXXFLAGS += -DVERIFICATION
endif

# Adds outer 'benchmarking' loop to do multiple trials for
# 1 < threads <= max_threads
ifeq ($(BENCHMARK),yes)
  CFLAGS += -DBENCHMARK
endif

# Binary dump for file I/O based initialization
ifeq ($(BINARY_DUMP),yes)
  CFLAGS += -DBINARY_DUMP
endif

# Binary read for file I/O based initialization
ifeq ($(BINARY_READ),yes)
  CFLAGS += -DBINARY_READ
endif


#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) XSbench_header.h
	$(CC) $(LDFLAGS) -o $@ $(obj) $(LDLIBS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

edit:
	vim -p $(source) papi.c XSbench_header.h

run:
	./$(program)

bgqrun:
	qsub -t 10 -n 1 -O test XSBench
