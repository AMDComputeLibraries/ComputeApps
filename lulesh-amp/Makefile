SHELL = /bin/sh

CXX = clang++

LULESH_EXEC = lulesh

INCLUDE_PATH=
LIB_PATH=

CXXFLAGS = `clamp-config --install --cxxflags` -g -Xclang -fhsa-ext -DBLOCKSIZE=128 
LDFLAGS = `clamp-config --install --ldflags` -lm
OPTS = -O3

ifeq "$(TIMING)" "TRUE"
	CXXFLAGS += -DKERNELS_TIMING
endif

SOURCES = \
	lulesh.cc \
	lulesh-util.cc \
	lulesh-init.cc \
	lulesh_kernels.cc
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
