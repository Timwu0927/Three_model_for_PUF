#!/bin/bash
#$ -l h_rt=6:00:00  #time needed
#$ -pe smp 10 #number of cores
#$ -l rmem=24G #number of memery
#$ -P rse-com6012
#$ -q rse-com6012.q
#$ -o ../Output/QP1.txt  #This is where your output and errors are logged.
#$ -j y # normal and error outputs into a single file (the file above)
#$ -M twu48@sheffield.ac.uk #Notify you by email, remove this line if you don't like
#$ -m ea #Email you when it finished or aborted
#$ -cwd # Run job from current directory

module load apps/java/jdk1.8.0_102/binary

module load apps/python/conda

source activate myspark

spark-submit --driver-memory 24g --executor-memory 24g --master local[5] ../assign1/QP1.py  # .. is a relative path, meaning one level upi

spark-submit --driver-memory 24g --executor-memory 24g --master local[10] ../assign1/QP1.py

