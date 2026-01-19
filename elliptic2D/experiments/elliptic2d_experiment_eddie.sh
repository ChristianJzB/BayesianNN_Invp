#!/bin/bash

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
KERNEL_SPATIAL="Matern52"   #Options "Matern52""Matern32" "SquaredExponential"
KERNEL_PARAMETER="Matern52"

#--- Model configurations ---

Ns=(10 20 30 40 50 75 100 200 300 400 500 1000 2000 3000 5000 7000 9000 10000 20000)
SMALL_BATCH=10                 # for N < 100
BIG_BATCHES=(200)     # for N ≥ 100 (you can add more)
NNEURONS=(80)
KLs=(2 3 4 5)

CONFIG_MODELS_DGALA=()

for KL in "${KLs[@]}"; do
  for N in "${Ns[@]}"; do

    # Decide which batch sizes should be used
    if (( N < 100 )); then
        batch_list=("$SMALL_BATCH")
    else
        batch_list=("${BIG_BATCHES[@]}")
    fi

    # Loop over batch sizes
    for BC in "${batch_list[@]}"; do
      for NN in "${NNEURONS[@]}"; do
        CONFIG_MODELS_DGALA+=("2 $NN $N $BC $KL")
      done
    done

  done
done

Ns=(1 2 3 4 5 6 7 8 9 10 15 20 25 30 35 40 45 50 60 65 75 100 125 150 160 170 180 190 200 210 220 230 240 250)
KLs=(2 3 4 5)

# Empty array
CONFIG_MODELS_PIGP=()
CONFIG_MODELS_GP=()

# Generate combinations
for KL in "${KLs[@]}"; do
  for N in "${Ns[@]}"; do
    CONFIG_MODELS_PIGP+=("$N $KL")
    CONFIG_MODELS_GP+=("$N $KL")
  done
done

# --- FEM configurations ---
CONFIG_MODELS_FEM=("2" "3" "4" "5")


# --- Loop over models ---
echo "Running experiments for model type: $MODEL_TYPE"

if [[ "$MODEL_TYPE" == "dgala" ]]; then
  CONFIGS=("${CONFIG_MODELS_DGALA[@]}")
  for CONFIG in "${CONFIGS[@]}"; do
    read HIDDEN_LAYERS NUM_NEURONS N BATCH_SIZE KL <<< "$CONFIG"
    echo "Submitting DGaLA job: N=$N, Layers=$HIDDEN_LAYERS, Neurons=$NUM_NEURONS, Batch=$BATCH_SIZE, KL=$KL"

qsub -N "ell2d_NN_N${N}" <<EOF
#!/bin/bash
#$ -cwd
# -q gpu
# -l gpu=1 
#$ -l h_rss=32G
#$ -l h_rt=36:00:00 

# Load necessary modules
. /etc/profile.d/modules.sh
module load cuda/12.1
module load miniforge
conda activate experiments

python elliptic2D/experiments/elliptic2d_models_experiment.py\
 --model_type $MODEL_TYPE $VERBOSE $DGALA $MCMC $DA $MARGINAL_APPROXIMATION $ALPHA_EVAL $PROPOSAL\
 --N $N\
 --hidden_layers $HIDDEN_LAYERS \
 --num_neurons $NUM_NEURONS \
 --batch_size $BATCH_SIZE \
 --kl $KL \
 --repeat $REPEAT
EOF
done

elif [[ "$MODEL_TYPE" == "gp" ]]; then
  CONFIGS=("${CONFIG_MODELS_GP[@]}")
  for CONFIG in "${CONFIGS[@]}"; do
    read N KL <<< "$CONFIG"
    echo "Submitting GP job: N=$N, KL=$KL"

qsub -N "ell2d_GP_N${N}" <<EOF
#!/bin/bash
#$ -cwd
# -q gpu
# -l gpu=1 
#$ -l h_rss=32G
#$ -l h_rt=36:00:00 

# Load necessary modules
. /etc/profile.d/modules.sh
module load cuda/12.1
module load miniforge
conda activate experiments

python elliptic2D/experiments/elliptic2d_models_experiment.py \
  --model_type $MODEL_TYPE $VERBOSE  $DGALA $MCMC $DA $MARGINAL_APPROXIMATION $ALPHA_EVAL $PROPOSAL \
  --kernel_parameter $KERNEL_PARAMETER\
  --N $N \
  --kl $KL \
  --repeat "$REPEAT"
EOF
  done

elif [[ "$MODEL_TYPE" == "pigp" ]]; then
  CONFIGS=("${CONFIG_MODELS_PIGP[@]}")
  for CONFIG in "${CONFIGS[@]}"; do
    read N KL <<< "$CONFIG"
    echo "Submitting PIGP job: N=$N, KL=$KL"

qsub -N "ell2d_PIGP_N${N}" <<EOF
#!/bin/bash
#$ -cwd
# -q gpu
# -l gpu=1 
#$ -pe sharedmem 2
#$ -l h_rss=32G
#$ -l h_rt=36:00:00 

# Load necessary modules
. /etc/profile.d/modules.sh
module load cuda/12.1
module load miniforge
conda activate experiments

python elliptic2D/experiments/elliptic2d_models_experiment.py \
  --model_type $MODEL_TYPE $VERBOSE  $DGALA $MCMC $DA $MARGINAL_APPROXIMATION $ALPHA_EVAL $PROPOSAL \
  --kernel_parameter $KERNEL_PARAMETER\
  --kernel_spatial $KERNEL_SPATIAL\
  --N $N \
  --kl $KL \
  --repeat $REPEAT
EOF
  done

elif [[ "$MODEL_TYPE" == "fem" ]]; then
  CONFIGS=("${CONFIG_MODELS_FEM[@]}")
  for CONFIG in "${CONFIGS[@]}"; do
    read KL <<< "$CONFIG"
    echo "Submitting FEM job:KL=$KL"

qsub -N "ell_FEM_KL${KL}" <<EOF
#!/bin/bash
#$ -cwd
# -q gpu
# -l gpu=1 
#$ -l h_rss=32G
#$ -l h_rt=36:00:00 

# Load necessary modules
. /etc/profile.d/modules.sh
module load cuda/12.1
module load miniforge
conda activate experiments

python elliptic2D/experiments/elliptic2d_models_experiment.py \
  --model_type $MODEL_TYPE $VERBOSE  $MCMC $PROPOSAL \
  --kernel_parameter $KERNEL_PARAMETER\
  --kernel_spatial $KERNEL_SPATIAL\
  --kl $KL \
  --repeat $REPEAT
EOF
  done

else
  echo "Error: Unsupported model type '$MODEL_TYPE'"
  exit 1
fi