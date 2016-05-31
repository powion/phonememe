#!/bin/bash -l
#SBATCH -J taligen16-proj
#SBATCH -A edu16.DT2118
#SBATCH -t 06:00:0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=BEGIN,END

source /usr/share/Modules/init/bash
source bashrc
set -e
exec python "$@"
