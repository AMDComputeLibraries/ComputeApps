#!/usr/bin/python
import os,sys
from argparse import Namespace
sys.path.insert(0, os.path.abspath('.'))
import hpgmgconf
hpgmgconf.configure(Namespace(CC='/usr/bin/gcc', CFLAGS='-fopenmp -O3 -march=bdver2 -mavx -ffast-math', CPPFLAGS='-DSTENCIL_fv2 -DAMP_NONE', LDFLAGS='', LDLIBS='', arch='build', fe=False, fv=True, fv_coarse_solver='bicgstab', fv_cycle='F', fv_mpi=False, fv_smoother='cheby', fv_subcomm=True, petsc_arch='', petsc_dir='', with_hpm=None))