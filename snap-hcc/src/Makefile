# This Makefile is for building GPU versions of SNAP.
# Use MODEL=LC (default) for discrete GPU,
# MODEL=HSAIL for APU;
# otherwise MODEL will be set to OLD 
# and we'll build for a pre-ROCM dGPU.

MODEL ?= LC
ifeq ($(MODEL),LC)
# dGPU version
$(info building for discrete GPU with ROCM)
CPPAMP_BUILD = /opt/rocm/hcc-lc
HCC_LC = true
else
ifeq ($(MODEL),HSAIL)
# APU version
$(info building for APU with ROCM)
CPPAMP_BUILD = /opt/rocm/hcc-hsail
HCC_LC = false
else
$(info building for pre-ROCM discrete GPU)
override MODEL=OLD
# Pre-ROCM version
CPPAMP_BUILD = /opt/hcc
HCC_LC = true
endif
endif

CLANG             = $(CPPAMP_BUILD)/bin/hcc
CLANGCFLAGS   = $(shell $(CPPAMP_BUILD)/bin/hcc-config --cxxflags)
ifeq ($(HCC_LC),true)
CLANGCFLAGS += -DUSE_HCC_LC
endif

CLANGLFLAGS   = $(shell $(CPPAMP_BUILD)/bin/hcc-config --ldflags)
CLANGFLAGS = -I. -g -O3 -march=native \
           $(CLANGCFLAGS) \
           -ferror-limit=1 \
           -I/usr/lib/openmpi/include

CLANGLNKFLAGS = $(CLANGLFLAGS) -g -lm



MPILIBS = -L/usr/lib/openmpi/lib -lmpi


SRCS = 	snap_main.cc \
	input.cc \
	allocate.cc \
	comms.cc \
	problem.cc \
	source.cc \
	sweep.cc \
	buffers.cc \
	scalar_flux.cc \
	convergence.cc \
	population.cc \
	profiler.cc

OBJS = $(SRCS:%.cc=objs/%.o)
all: objs snap

objs:
	mkdir objs

snap: $(OBJS)
	$(CLANG) $(LDFLAGS) $(CLANGLNKFLAGS) $(MPILIBS) $^ -o $@

objs/snap_main.o: snap_main.cc hcc_arrays.h hcc_planes.h
	$(CLANG) $(CLANGFLAGS) -c snap_main.cc -o $@

objs/%.o: %.cc
	$(CLANG) $(CLANGFLAGS) -c $< -o $@


.PHONY: clean

clean:
	rm -f snap objs/*o
