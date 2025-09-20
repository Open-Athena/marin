## Overview

PlantCAD1 reproduction experiments.

## Setup

Original tutorial: https://gist.github.com/eric-czech/31e5b79689d322f7becb94a109ce0b75

### Local

```bash
git clone https://github.com/marin-community/marin.git
cd marin
uv venv --python 3.11
uv sync
```

### SkyPilot

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
uv pip install "skypilot[lambda]==0.10.3"
sky check lambda
sky launch \
  --cluster marin --num-nodes 1 --gpus "A10:1" --disk-size 100 \
  --env HUGGING_FACE_HUB_TOKEN --env WANDB_API_KEY \
  output/cluster.sky.yaml --retry-until-up --yes
rsync -rPz ./ marin:/home/ubuntu/sky_workdir --exclude '.venv' --exclude '.git' --exclude src/marin/markdown
```

#### CoreWeave

```bash
uv pip install "skypilot[kubernetes]==0.10.3"
sky check k8s
sky launch \
  --cluster marin --num-nodes 1 --infra k8s --gpus "H100_NVLINK_80GB:8" \
  --cpus 124 --memory 2008 \
  --env HUGGING_FACE_HUB_TOKEN --env WANDB_API_KEY \
  output/cluster.sky.yaml
rsync -rPz ./ marin:/home/sky/sky_workdir --exclude '.venv' --exclude '.git' --exclude src/marin/markdown

# For transformer-engine-jax:
sudo apt update
sudo apt install build-essential g++ cmake ninja-build
# uv sync --extra cuda12
# hint: This error likely indicates that you need to install a library that provides "cuda_runtime_api.h" for `transformer-engine-jax@2.6.0.post1`
```

#### Run

```bash
ssh marin
cd sky_workdir && conda deactivate && source .venv/bin/activate
export RAY_DEBUG=legacy

python -m experiments.plantcad.exp_pc1_tutorial --prefix local_store --force_run_failed true
python -m experiments.plantcad.exp_pc1_batch_tune --prefix local_store --force_run_failed true

python -m experiments.plantcad.exp_pc1_lr_tune --prefix local_store --force_run_failed true
find local_store | grep -E 'step-668$' | xargs -I {} echo "hf upload plantcad/_dev_marin_plantcad1_v1_lr_tune {} {} --repo-type model"

python -m experiments.plantcad.exp_pc1_eval --prefix local_store --force_run_failed true
```

## EDA 

### Tokenizer stats

From https://huggingface.co/kuleshov-group/PlantCaduceus_l20, e.g.:
PlantCaduceus vocab size: 7

### Dataset stats
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

This means 2,808,464,384 / 20 ==> ~140.4M params is Chinchilla optimal for text.

## TODO

- Look for prefetch config
- Debug: "Your setup doesn't support bf16/gpu." in eval with `bf16_full_eval`

```
# cat /tmp/ray/session_2025-09-20_04-11-22_232072_15326/runtime_resources/pip/f20b7e798eeb2fc9320b1a708aaeee4e0130ee14/virtualenv/lib/python3.11/site-packages/transformers/training_args.py | grep -i "doesn't support" -C 100
if self.bf16 or self.bf16_full_eval:
    if self.use_cpu and not is_torch_available() and not is_torch_xla_available():
        # cpu
        raise ValueError("Your setup doesn't support bf16/(cpu, tpu, neuroncore). You need torch>=1.10")
    elif not self.use_cpu:
        if not is_torch_bf16_gpu_available() and not is_torch_xla_available():  # added for tpu support
            error_message = "Your setup doesn't support bf16/gpu."
            if is_torch_cuda_available():
                error_message += " You need Ampere+ GPU with cuda>=11.0"
            # gpu
            raise ValueError(error_message)
```

- Discuss: `levanter.data.loader - loader.py:258 - INFO :: Prefetch wasn't fast enough: 33.836.`
- Discuss this:

```
# TODO: discuss https://github.com/jax-ml/jax/issues/24909
# (train_lm_task pid=31054) /tmp/ray/session_2025-09-16_11-16-05_933535_22116/runtime_resources/pip/96e8d2e31c1b75b4d19a0ea2c755a672438fdca3/virtualenv/lib/python3.11/site-packages/levanter/layers/attention.py:428: UserWarning: transformer_engine is not installed. Please install it to use NVIDIA's optimized fused attention.. Falling back to the reference implementation.
# (train_lm_task pid=31054)   warnings.warn(f"{msg}. Falling back to the reference implementation.")
# (train_lm_task pid=31054) E0916 11:23:11.594742   31054 buffer_comparator.cc:150] Difference at 10780: 16.375, expected 14.5
# (train_lm_task pid=31054) E0916 11:23:11.594787   31054 buffer_comparator.cc:150] Difference at 10942: 17.25, expected 15.25
# (train_lm_task pid=31054) E0916 11:23:11.594791   31054 buffer_comparator.cc:150] Difference at 11042: 17, expected 15.1875
# (train_lm_task pid=31054) E0916 11:23:11.594795   31054 buffer_comparator.cc:150] Difference at 11132: 16.875, expected 14.8125
# (train_lm_task pid=31054) E0916 11:23:11.594801   31054 buffer_comparator.cc:150] Difference at 12211: 15, expected 16.875
# (train_lm_task pid=31054) E0916 11:23:11.594804   31054 buffer_comparator.cc:150] Difference at 12212: 14.625, expected 16.625
# (train_lm_task pid=31054) E0916 11:23:11.594807   31054 buffer_comparator.cc:150] Difference at 12235: 14.75, expected 16.625
# (train_lm_task pid=31054) E0916 11:23:11.594809   31054 buffer_comparator.cc:150] Difference at 12276: 15.0625, expected 16.875
# (train_lm_task pid=31054) E0916 11:23:11.594812   31054 buffer_comparator.cc:150] Difference at 12327: 14.5, expected 16.25
# (train_lm_task pid=31054) E0916 11:23:11.594815   31054 buffer_comparator.cc:150] Difference at 12336: 15.5625, expected 17.5
# (train_lm_task pid=31054) 2025-09-16 11:23:11.594824: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1070] Results do not match the reference. This is likely a bug/unexpected loss of precision.
```