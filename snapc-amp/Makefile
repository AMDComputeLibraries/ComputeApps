A_PREFIX=snap
A_SUFFIX=_c
A_TARGET=$(A_PREFIX)$(A_SUFFIX)

CC=mpicc			# C compiler command
CFLAGS=-g -O3 -std=c99   -march=bdver2 	# C compiler flags
#CC=gcc
#CFLAGS=-g -O3 -std=c99 -fopenmp  #-wall	# C compiler flags
LDFLAGS= -g -lm 		# Linker flags, e.g. -L<lib dir> if you have libraries in a nonstandard directory <lib dir>
LIBS=				# Libraries to pass to the linker, e.g. -l<library>
CPPFLAGS=			# (Objective) C/C++ preprocessor flags, e.g. -I<include dir> if you have headers in a nonstandard directory <include dir>
CXX=g++				# C++ compiler command
CXXFLAGS=-g -O3 -std=c++11 	# C++ compiler flags
CXXCPP=				# C++ preprocessor flags

CPPAMP_BUILD = /opt/hcc
CLANG             = $(CPPAMP_BUILD)/bin/clang++
CLANGFLAGS = -I. -O3 -march=bdver2 -g\
           -DHAVE_AMP \
           -I$(CPPAMP_BUILD)/include \
           -hc \
           -std=c++amp \
           -stdlib=libc++ \
           -Xclang \
           -fhsa-ext -DCXXAMP_ENABLE_HSA=1 \
           -I/usr/lib/openmpi/include
CLANGLNKFLAGS = -hc -std=c++amp -L$(CPPAMP_BUILD)/lib -Wl,--rpath=$(CPPAMP_BUILD)/lib -lc++ -lc++abi -ldl -Wl,--whole-archive -lmcwamp -Wl,--no-whole-archive
MPILIBS = -L/usr/lib/openmpi/lib -lmpi
LIBS+=$(MPILIBS)

A_FILES := $(wildcard *.cpp) $(wildcard *.c)
A_OBJS := $(sort $(patsubst %.c, %.o, $(patsubst %.cpp, %.o, $(A_FILES))))

INPUT_FILE=scale_test_8_4.inp
OUTPUT_PREFIX=snap_test
OUTPUT_SUFFIX=.out
OUTPUT_FILE=$(OUTPUT_PREFIX)$(OUTPUT_SUFFIX)

SRCS = snap.h snap_main.c utils.c version.c plib.c geom.c \
       sn.c data.c control.c input.c setup.c dealloc.c \
       translv.c solvar.c outer.c expxs.c inner.c sweep.c \
       octsweep.c dim1_sweep.c dim3_sweep.c output.c time.c mms.c


all: $(A_TARGET) 			# Build all targets

vars:					# Show Makefile variable values
	@echo "       CC "$(CC)
	@echo " A_TARGET "$(A_TARGET)
	@echo "   CFLAGS "$(CFLAGS)
	@echo "  LDFLAGS "$(LDFLAGS)
	@echo "     LIBS "$(LIBS)@egrep "[a-z]\w+:" Makefile | sort

clean:					# Remove make build object and target files
	-rm -f *o *~ $(A_TARGET)

cleanout: clean
	-rm -f $(OUTPUT_FILE)

.PHONY:	clean vars help install rebuild

rebuild: clean $(A_TARGET)		# Clean and build target

install: 				# Build and install

run: $(A_TARGET)
	./$(A_TARGET) --fi $(INPUT_FILE) --fo $(OUTPUT_FILE)

$(A_TARGET): $(A_OBJS)			# Build target and its required object files
	$(CLANG)  $^ -o $@ $(LDFLAGS) $(CLANGLNKFLAGS) $(LIBS)
#	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(LIBS)

dim3_sweep.o: dim3_sweep.cpp
	$(CLANG) $(CLANGFLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(CXXCPP) -c $< -o $@

%: %.cpp
	$(CXX) $(CXXFLAGS) $(CXXCPP) -o $@ $<

%.o: %.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

%: %.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ $<

#
# Count lines of code
#
count:
	rm -f Lines
	for file in $(SRCS); do ./LineCount $$file Lines; done
	gawk -f ./LineReport < Lines >> Lines
	cat Lines
