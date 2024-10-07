#!/bin/bash
#SBATCH -J mas-%j
#SBATCH --mail-type=Fail
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1

#SBATCH -t 24:00:00
#SBATCH --mem=8000
#SBATCH -C XEON_SP_6126

set -e
echo "$@"
python "$@"
