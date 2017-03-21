#!/bin/bash
#
#PBS -q interactive
#PBS -N dwdiij1
#p` –lselect=1:ncpus=1
#PBS -l place=free
#PBS -V

# change to the working directory
cd $PBS_O_WORKDIR

echo ">>>> Begin j1"
ldd --version > lddv.out

module load python/2.7.13_anaconda
modeul load glib/2.50

# actual binary (with IO redirections) and required input
# parameters is called in the next line
python job.py  > j1.out  2>&1

echo ">>>> Begin j1 Run ..."
