#!/bin/bash
#SBATCH --account=nn2806k
#SBATCH --time=04:00:00
#SBATCH --job-name=mlp-hp-tune
## You can add more args for sbatch here in the same way as above (see "sbatch --help" for all the args), e.g.:
#SBATCH --ntasks=10
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=10

# Exit the script on any error
set -o errexit

cd /cluster/work/users/lassetk/dm-vegetation-types-norway/notebooks

# Load SAGA modules and conda env
. ../load_saga_dependencies.sh

# Run analysis
python3 H-hyperparameter_tuning_full.py --classifier mlp --n_iterations 10 --use-log-file
