#! /bin/bash
#
#PBS -l walltime=00:10:00
#PBS -l nodes=1:ppn=16
#PBS -W group_list=newriver
#PBS -q open_q
#PBS -j oe

cd $PBS_O_WORKDIR

module purge
module load gcc

make

for Nthreads in `seq 1 16`
do 
    ./mandelbrot 4096 4096 $Nthreads;
done;



