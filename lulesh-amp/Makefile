SHELL = /bin/sh

#Array_view mode
HCC_ARR_VIEW = ON

# HSA Machine
#HCC_PATH=/opt/rocm/hcc-hsail

# dGPU Machine
HCC_PATH=/opt/rocm/hcc

HCC_CONFIG=$(HCC_PATH)/bin/hcc-config
CXX = $(HCC_PATH)/bin/hcc

LULESH_EXEC = lulesh

INCLUDE_PATH=
LIB_PATH=

CXXFLAGS = $(shell $(HCC_CONFIG) --install --cxxflags) -DBLOCKSIZE=256
LDFLAGS = $(shell $(HCC_CONFIG) --install --ldflags) -lm

ifeq ($(HCC_ARR_VIEW), ON)
  CXXFLAGS += -DARRAY_VIEW
endif


OPTS = -O3

SOURCES = \
	lulesh.cc \
	lulesh-util.cc \
	lulesh-init.cc

OBJECTS = $(SOURCES:%.cc=objs/%.o)

.SUFFIXES: .cc .o

all: CXXFLAGS += $(OPTS)
all: LDFLAGS += $(OPTS)
all: $(LULESH_EXEC)

sp: CXXFLAGS += -DSINGLE
sp: CXXFLAGS += $(OPTS)
sp: LDFLAGS += $(OPTS)
sp: $(LULESH_EXEC)

debug: CXXFLAGS += -g -DDEBUG
debug: $(LULESH_EXEC)

$(OBJECTS): | objs

objs:
	@mkdir -p $@

objs/%.o: %.cc
	@echo "Building $<"
	$(CXX) -c $(CXXFLAGS) -o $@  $<

lulesh: $(OBJECTS)
	@echo "Linking"
	$(CXX) $(OBJECTS) $(LDFLAGS) -lm -o $@

clean:
	/bin/rm -rf *~ objs/ $(LULESH_EXEC)
	/bin/rm -rf *.dSYM

tar: clean
	cd .. ; tar cjvf luleshCL.tar.bz2 LULESH/*; mv luleshCL.tar.bz2 LULESH
