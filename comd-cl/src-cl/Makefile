# Makefile for src-parallel

SHELL = /bin/bash

### CoMD can be built in either double or single precision and with or
### without MPI.  Select desired precision and MPI here.

# double precision (ON/OFF)
DOUBLE_PRECISION = ON
# MPI for parallel (ON/OFF)
DO_MPI = OFF

### Set your desired C compiler and any necessary flags.  Note that CoMD
### uses some c99 features.  You can also set flags for optimization and
### specify paths to include files that the compiler can't find on its
### own.  If you need any -L or -l switches to get C standard libraries
### (such as -lm for the math library) put them in C_LIB.
CC = gcc
CFLAGS = -std=c99
OPTFLAGS = -g -O5
INCLUDE_PATH = /opt/AMDAPP/include
INCLUDES = -I$(INCLUDE_PATH) 
#C_LIB = -lm -framework OpenCL
C_LIB = -lm -lOpenCL


### If you need to specify include paths, library paths, or link flags
### for MPI, put them here.  Put both -L and -l switches into MPI_LIB.
MPI_LIB =
MPI_INCLUDE =

### A place to specify any other include or library switches your
### platform requires.
OTHER_LIB =
OTHER_INCLUDE =

#########################################
### Below here, it is pitch black.  
### You are likely to be eaten by a grue.
##########################################

# clear all suffixes
.SUFFIXES:
# list only those that we use 
.SUFFIXES: .c .o

.PHONY: DEFAULT clean distclean depend

# Check for double precision
ifeq ($(DOUBLE_PRECISION), ON)
CFLAGS += -DDOUBLE
else
CFLAGS += -DSINGLE
endif

# Add includes & libraries for MPI if needed.
ifeq ($(DO_MPI), ON)
INCLUDES += ${MPI_INCLUDE}
CFLAGS += -DDO_MPI
LDFLAGS += ${MPI_LIB}
endif

LDFLAGS += ${C_LIB} ${OTHER_LIB}
CFLAGS  += ${OPTFLAGS} ${INCLUDES} ${OTHER_INCLUDE}


SOURCES=$(wildcard *.c)
OBJECTS=$(SOURCES:.c=.o)


DEFAULT: ../CoMD-ocl

%.o: %.c
	${CC} ${CFLAGS} -c $< -o $@

../CoMD-ocl: CoMD_info.h ${OBJECTS} 
	${CC} ${CFLAGS} -o ../CoMD-ocl ${OBJECTS} ${LDFLAGS}

CoMD_info.h: Makefile
	./generate_info_header "$(CC)" "$(CFLAGS)" "$(LDFLAGS)"

clean:
	rm -f *.o CoMD_info.h .depend

distclean: clean
	rm -f ../CoMD-ocl
	rm -rf html latex

.depend: $(SOURCES)
	@touch .depend
	@$(MAKE) --no-print-directory depend

depend:
	@echo "Rebuilding dependencies..."
	@$(MAKE) CoMD_info.h
	@makedepend -f .depend -Y. --$(CFLAGS)-- $(SOURCES) 2> /dev/null


#-include .depend
