#===============================================================================
# User Options
#===============================================================================

COMPILER    = clang
OPTIMIZE    = no
DEBUG       = no
PROFILE     = no
MPI         = no
PAPI        = no
VEC_INFO    = no
VERIFY      = no
BENCHMARK   = no
BINARY_DUMP = no
BINARY_READ = no

#===============================================================================
# Program name & source code list
#===============================================================================

program = XSBench

source = \
Main.c \
io.c \
CalculateXS.c \
GridInit.c \
XSutils.c \
Materials.c

obj = $(source:.c=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := -std=gnu99

# Linker Flags
LDFLAGS = -lm

# Regular gcc Compiler
ifeq ($(COMPILER),gnu)
  CC = gcc
  CXX = g++
  CFLAGS += -fopenmp
  CXXFLAGS = -fopenmp
  LDFLAGS += $(CFLAGS)
endif

# C++AMP clang Compiler
ifeq ($(COMPILER),clang)
  CLAMP_HOME = /opt/hcc
  INCLUDES = -I.

  CC = $(CLAMP_HOME)/bin/clang
  CXX = $(CLAMP_HOME)/bin/clang++ -std=c++amp
  LDFLAGS = $(shell $(CLAMP_HOME)/bin/clamp-config --install --ldflags) -lc -lc++
  LDLIBS  =  -lpthread -lm -lmpi -ldl -lelf

  CXXFLAGS = $(shell $(CLAMP_HOME)/bin/clamp-config --install --cxxflags) \
           -O3 \
           -DHAVE_AMP \
	   -Xclang \
	   -fhsa-ext

  CFLAGS = -O3 \
           -DHAVE_AMP

endif

# Intel Compiler
ifeq ($(COMPILER),intel)
  CC = icc
  CFLAGS += -openmp 
endif

# BG/Q gcc Cross-Compiler
ifeq ($(MACHINE),bluegene)
  CC = mpicc
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
	$(CXX) $(LDFLAGS) -o $@ $(obj) $(LDLIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX) $(INCLUDES) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

edit:
	vim -p $(source) papi.c XSbench_header.h

run:
	./$(program)

bgqrun:
	qsub -t 10 -n 1 -O test XSBench
