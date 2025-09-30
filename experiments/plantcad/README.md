## Overview

PlantCAD1 reproduction experiments.

Original tutorial: https://gist.github.com/eric-czech/31e5b79689d322f7becb94a109ce0b75

## Setup

### Local

```bash
git clone https://github.com/marin-community/marin.git
cd marin
uv venv --python 3.11
uv sync
```

### Remote (SkyPilot)

```bash
sky api stop; [ -d ~/.sky ] && rm -rf ~/.sky

mkdir -p output

cat << 'EOF' > output/cluster.sky.yaml
envs:
  HUGGING_FACE_HUB_TOKEN: null
  WANDB_API_KEY: null
workdir: .
setup: |
  uv venv --python 3.11
  uv sync --extra=cuda12
  for var in HUGGING_FACE_HUB_TOKEN WANDB_API_KEY; do
    declare -n ref=$var
    grep -q "^export $var=" ~/.bashrc || echo "export $var=$ref" >> ~/.bashrc
  done
EOF
```

#### Lambda

```bash
uv pip install "skypilot[lambda]==0.10.3.post1"
sky check lambda
sky launch \
  --cluster marin --infra lambda --num-nodes 1 --gpus "A100:8" --disk-size 100 \
  --env HUGGING_FACE_HUB_TOKEN --env WANDB_API_KEY \
  output/cluster.sky.yaml --retry-until-up --yes
REMOTE_USER=ubuntu
```

#### GCP

```bash
uv pip install "skypilot[gcp]==0.10.3.post1"
sky check gcp
sky launch \
  --cluster marin --infra gcp --num-nodes 1 --gpus "A100:1" --disk-size 100 \
  --instance-type a2-highgpu-1g --region us-east-1 \
  --env HUGGING_FACE_HUB_TOKEN --env WANDB_API_KEY \
  output/cluster.sky.yaml
REMOTE_USER=gcpuser
```

#### CoreWeave

```bash
# The default timeout for pod launch is too conservative in SkyPilot and needs to be increased:
mkdir -p ~/.sky; cat > ~/.sky/config.yaml << EOF
kubernetes:
  provision_timeout: 180 # Wait 3 minutes for provisioning before timeout
  autoscaler: coreweave
EOF

uv pip install "skypilot[kubernetes]==0.10.3.post1"
sky check k8s
sky show-gpus --infra k8s
sky launch \
  --cluster marin --num-nodes 1 --infra k8s --gpus "H100_NVLINK_80GB:8" \
  --cpus 124 --memory 2008 \
  --env HUGGING_FACE_HUB_TOKEN --env WANDB_API_KEY \
  output/cluster.sky.yaml
REMOTE_USER=sky

# For transformer-engine-jax:
sudo apt update
sudo apt install build-essential g++ cmake ninja-build
# uv sync --extra cuda12
# hint: This error likely indicates that you need to install a library that provides "cuda_runtime_api.h" for `transformer-engine-jax@2.6.0.post1`

# For a manual debugging pod:
kubectl get nodes -o wide # get name "gd92c2c"
kubectl debug node/gd92c2c -i -t --image=ubuntu
```

## Execution

```bash
ssh marin
cd sky_workdir && conda deactivate && source .venv/bin/activate
export RAY_DEBUG=legacy

# Code sync
rsync -rPz ./ marin:/home/$REMOTE_USER/sky_workdir \
  --exclude '.venv' --exclude '.git' \
  --exclude src/marin/markdown --exclude '__pycache__'

# Experiments and tuning
python -m experiments.plantcad.scripts.exp_pc1_tutorial --prefix local_store --force_run_failed true
python -m experiments.plantcad.scripts.exp_pc1_batch_tune --prefix local_store --force_run_failed true
python -m experiments.plantcad.scripts.exp_pc1_lr_tune --prefix local_store --force_run_failed true

# Training
sudo apt-get install screen -y; screen -S train
mkdir -p logs
python -m experiments.plantcad.scripts.exp_pc1_train \
  --prefix local_store --force_run_failed true 2>&1 | tee logs/exp_pc1_train.log

# Evaluation
rm -rf local_store/evaluation/dna-conservation*; python -m experiments.plantcad.scripts.exp_pc1_eval --prefix local_store --force_run_failed true

# Checkpoint upload
find local_store | grep -E 'hf/step-[0-9]+$' | xargs -I {} echo "hf upload plantcad/_dev_marin_plantcad1_v2_train {} {} --repo-type model" | bash /dev/stdin
```

```bash
> python -m experiments.plantcad.misc.agg_eval_results
roc_auc  step                                                                                                  checkpoint_path
0.535217  1673  hf://plantcad/_dev_marin_plantcad1_v1_train/local_store/checkpoints/plantcad-train-300m-r02-432442/hf/step-1673
0.546725  3346  hf://plantcad/_dev_marin_plantcad1_v1_train/local_store/checkpoints/plantcad-train-300m-r02-432442/hf/step-3346
0.549917  5019  hf://plantcad/_dev_marin_plantcad1_v1_train/local_store/checkpoints/plantcad-train-300m-r02-432442/hf/step-5019
0.558042  6692  hf://plantcad/_dev_marin_plantcad1_v1_train/local_store/checkpoints/plantcad-train-300m-r02-432442/hf/step-6692
0.560290  8365  hf://plantcad/_dev_marin_plantcad1_v1_train/local_store/checkpoints/plantcad-train-300m-r02-432442/hf/step-8365
0.565785 10038 hf://plantcad/_dev_marin_plantcad1_v1_train/local_store/checkpoints/plantcad-train-300m-r02-432442/hf/step-10038
0.570048 11711 hf://plantcad/_dev_marin_plantcad1_v1_train/local_store/checkpoints/plantcad-train-300m-r02-432442/hf/step-11711
0.576358 13384 hf://plantcad/_dev_marin_plantcad1_v1_train/local_store/checkpoints/plantcad-train-300m-r02-432442/hf/step-13384
0.583593 15057 hf://plantcad/_dev_marin_plantcad1_v1_train/local_store/checkpoints/plantcad-train-300m-r02-432442/hf/step-15057
0.585834 16730 hf://plantcad/_dev_marin_plantcad1_v1_train/local_store/checkpoints/plantcad-train-300m-r02-432442/hf/step-16730
0.589215 18403 hf://plantcad/_dev_marin_plantcad1_v1_train/local_store/checkpoints/plantcad-train-300m-r02-432442/hf/step-18403
0.588738 20076 hf://plantcad/_dev_marin_plantcad1_v1_train/local_store/checkpoints/plantcad-train-300m-r02-432442/hf/step-20076
0.593178 21749 hf://plantcad/_dev_marin_plantcad1_v1_train/local_store/checkpoints/plantcad-train-300m-r02-432442/hf/step-21749
```

Second iteration:

```
 python experiments/plantcad/misc/agg_eval_results.py
 roc_auc  step                                                                                                  checkpoint_path
0.549341  2678  hf://plantcad/_dev_marin_plantcad1_v2_train/local_store/checkpoints/plantcad-train-600m-r12-7ea0fc/hf/step-2678
0.566597  5356  hf://plantcad/_dev_marin_plantcad1_v2_train/local_store/checkpoints/plantcad-train-600m-r12-7ea0fc/hf/step-5356
0.604521  8034  hf://plantcad/_dev_marin_plantcad1_v2_train/local_store/checkpoints/plantcad-train-600m-r12-7ea0fc/hf/step-8034
0.626729 10712 hf://plantcad/_dev_marin_plantcad1_v2_train/local_store/checkpoints/plantcad-train-600m-r12-7ea0fc/hf/step-10712
0.631095 13390 hf://plantcad/_dev_marin_plantcad1_v2_train/local_store/checkpoints/plantcad-train-600m-r12-7ea0fc/hf/step-13390
```

## EDA

Stats on kuleshov-group/Angiosperm_16_genomes:

```
> python count_dataset.py
Number of examples: 5,485,282
Tokens per example: 512
Total tokens: 2,808,464,384
Total tokens (billions): 2.81B

Analyzing token frequencies from ENTIRE dataset...
Reading all 2,808,464,384 tokens directly from tensorstore...
Shape: (2808464384,), dtype: int64
Counting token frequencies with np.unique...

Token frequency analysis (FULL dataset - 2,808,464,384 tokens):
Most common tokens:
  Token 3: 845,215,196 occurrences (30.10%)
  Token 6: 845,194,154 occurrences (30.09%)
  Token 4: 558,408,742 occurrences (19.88%)
  Token 5: 558,387,426 occurrences (19.88%)
  Token 2: 1,258,866 occurrences (0.04%)

Total unique tokens: 5
Token ID range: 2 - 6
```
