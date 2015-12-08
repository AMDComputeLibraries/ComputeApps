SHELL = /bin/sh

CXX = g++

LULESH_EXEC = lulesh

INCLUDE_PATH=/opt/AMDAPP/include/
LIB_PATH=/opt/AMDAPP/lib/x86_64/

CXXFLAGS = -std=c++11 -Wall -I$(INCLUDE_PATH) 
LDFLAGS = -std=c++11 -L$(LIB_PATH) -lOpenCL
OPTS = -O3

ifeq "$(TIMING)" "TRUE"
	CXXFLAGS += -DKERNELS_TIMING
endif

SOURCES = \
	lulesh.cc \
	lulesh-util.cc \
	lulesh-init.cc \
	CLsetup.cc
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
