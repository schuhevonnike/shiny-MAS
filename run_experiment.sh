#!/bin/bash
#SBATCH -J mas-%j
#SBATCH --mail-type=Fail
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1

#SBATCH -t 24:00:00
#SBATCH --mem=15000
#SBATCH -C EPYC_7262

set -e
echo "$@"
python "$@"
