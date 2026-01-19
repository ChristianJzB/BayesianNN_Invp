#!/bin/bash

# Define different values of N for experiments
N_VALUES=(20 125 500 1000 1500 2000)  # Adjust as needed
N_LAYERS=(3)  # Define hidden layers
N_BATCHES=(160 320 480)
KL_VALUES=(1) 

# --- Configuration ---
VERBOSE="--verbose"
MODEL_TYPE="dgala"  # Options: dgala, gp, pigp
MARGINAL_APPROXIMATION=""
PROPOSAL="--proposal random_walk"
MCMC="--mcmc"
DA="--da"
ALPHA_EVAL="--alpha_eval"
REPEAT=3
DEEPGALA="--dgala"

for N in "${N_VALUES[@]}"; do
    for L in "${N_LAYERS[@]}"; do
        for BATCH in "${N_BATCHES[@]}"; do
            for KL in "${KL_VALUES[@]}"; do

            echo "Submitting job for N=$N with $L hidden layers"

qsub -N "nv_N${N}_L${L}_BS${BS}" <<EOF
#!/bin/bash
#$ -cwd
#$ -q gpu
#$ -l gpu=1 
# -pe sharedmem 4
#$ -l h_rss=32G
#$ -l h_rt=24:00:00 

# Load necessary modules
. /etc/profile.d/modules.sh
module load cuda/12.1
module load miniforge
conda activate experiments

# Run the experiment with dynamic and fixed arguments
python navier_stokes/experiments/nv_experiment.py\
 --model_type $MODEL_TYPE $VERBOSE $DGALA $MCMC $DA $MARGINAL_APPROXIMATION $ALPHA_EVAL $PROPOSAL\
 --N $N\
 --hidden_layers $L \
 --num_neurons 300 \
 --batch_size $BATCH \
 --kl $KL \
 --repeat $REPEAT
EOF
            done
        done
    done
done
