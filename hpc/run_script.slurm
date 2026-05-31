#!/bin/bash

#SBATCH --time=72:00:00                # Maximum execution time (hh:mm:ss)
#SBATCH --gpus-per-node=1              # Number of GPUs per node
#SBATCH --cpus-per-task=1              # Number of CPU cores per task
#SBATCH --constraint=[p100|v100|t4]    # GPU types (P100, V100, or T4)
#SBATCH --mem-per-cpu=80G              # Memory per CPU core
#SBATCH --job-name=deepcrime_task      # Job name
#SBATCH --mail-user=your_email@domain.com   # Email for notifications
#SBATCH --mail-type=ALL                # Notify on job start, end, and fail
#SBATCH --account=your_account_name    # Compute Canada account

# Record the start time
start_time=$(date +%s)

# Load the required Python module
module load python/3.10

# Create and activate a virtual environment in the temporary directory
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

# Install dependencies
pip install numpy tensorflow==2.15.1 matplotlib progressbar scikit-learn \
            termcolor h5py pandas statsmodels networkx patsy scipy psutil --no-index

# Running any scripts (e.g., run the DeepCrime script from Fault Seeding)
python run_deepcrime.py FNN_72328867_correct.py change_dropout_rate

# Record the end time and calculate duration
end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(( (duration % 3600) / 60 ))
seconds=$((duration % 60))

# Display runtime
echo "The script took $hours hours, $minutes minutes, and $seconds seconds to run."

# Deactivate the virtual environment
deactivate
