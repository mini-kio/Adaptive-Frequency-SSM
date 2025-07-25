# Adaptive-Frequency-SSM

An implementation of **Adaptive-Frequency State Space Models** with advanced frequency domain compression, based on the S6 architecture. This model aims to enable efficient long sequence modeling through spectral domain state compression techniques.

## About

Adaptive-Frequency-SSM implements advanced frequency domain compression for state space models, aiming for efficient processing of long sequences while maintaining performance. Key features include:

- **Frequency Domain State Compression**: Compresses state representations in the spectral domain
- **Adaptive Frequency Selection**: Learns which frequency components are most important
- **Multi-Scale Processing**: Handles multiple temporal resolutions simultaneously
- **S6-Based Architecture**: Built on the proven S6 state space model foundation
- **Spectral Regularization**: Ensures training stability through spectral normalization

## Features

- **Efficient Long Sequence Modeling**: Designed for long sequence processing up to 2048 tokens
- **Competitive Performance**: Designed for competitive results on LRA benchmark
- **Fast Training**: Optimized for mixed precision and distributed training
- **Flexible Architecture**: Support for various downstream tasks
- **Comprehensive Evaluation**: Built-in LRA benchmark evaluation
- **Distributed Training**: Multi-GPU and multi-node support
- **Experiment Tracking**: Integrated W&B logging

## Installation

### Prerequisites

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA-compatible GPU (recommended)

### Install from Source

```bash
git clone https://github.com/mini-kio/adaptive-frequency-ssm.git
cd adaptive-frequency-ssm
pip install -e .
```

### Install Dependencies

```bash
pip install torch numpy einops
pip install datasets transformers
pip install wandb tqdm
pip install matplotlib seaborn scikit-learn
```

### Development Installation

```bash
pip install -e ".[dev]"
```

## Usage

### Quick Start

```python
from adaptive_frequency_ssm import AdaptiveFrequencySSM, AdaptiveFrequencySSMConfig

# Create model configuration
config = AdaptiveFrequencySSMConfig(
    d_model=512,
    n_layer=12,
    vocab_size=50304,
    compression_ratio=0.5,
    use_adaptive_mask=True
)

# Initialize model
model = AdaptiveFrequencySSM(config, task="classification")

# Forward pass
input_ids = torch.randint(0, 1000, (32, 2048))  # (batch, seq_len)
outputs = model(input_ids=input_ids)
```

### Training

#### Single GPU Training

```bash
python -m adaptive_frequency_ssm.train \
    --task_name listops \
    --d_model 512 \
    --n_layer 12 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --max_steps 100000 \
    --compression_ratio 0.5 \
    --data_dir ./data \
    --save_dir ./checkpoints
```

#### Distributed Training

```bash
torchrun --nproc_per_node=4 -m spectral-ssm.train \
    --distributed \
    --task_name listops \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --data_dir ./data \
    --save_dir ./checkpoints
```

#### Configuration File Training

```bash
python -m adaptive_frequency_ssm.train --config configs/base_config.yaml
```

### Evaluation

#### LRA Benchmark Evaluation

```bash
python -m adaptive_frequency_ssm.evaluation \
    --checkpoint_path ./checkpoints/best_model.pt \
    --data_dir ./data \
    --output_dir ./evaluation_results \
    --tasks listops text retrieval pathfinder \
    --measure_throughput \
    --measure_memory
```

#### Performance Analysis

```bash
python -m adaptive_frequency_ssm.evaluation \
    --checkpoint_path ./checkpoints/best_model.pt \
    --profile_layers \
    --compression_ratios 0.25 0.5 0.75 1.0
```

### Aggregate Results

```bash
python scripts/aggregate_lra_results.py \
    --results_dir ./evaluation_results \
    --output_file ./final_results.json
```

## Configuration

### Model Configuration

```yaml
model:
  name: "adaptive_frequency_ssm_base"
  d_model: 512
  n_layer: 12
  vocab_size: 50304
  d_state: 64
  d_conv: 4
  expand: 2
  compression_ratio: 0.5
  use_adaptive_mask: true
  use_spectral_norm: true
  use_multi_scale: true
  num_scales: 3
  dropout: 0.1
  max_seq_len: 2048
```

### Training Configuration

```yaml
training:
  batch_size: 32
  max_steps: 100000
  learning_rate: 1e-4
  weight_decay: 0.01
  warmup_steps: 1000
  lr_decay_steps: 80000
  use_amp: true
  grad_clip_norm: 1.0
```

### Key Parameters

- **compression_ratio**: Controls frequency domain compression (0.0-1.0)
- **use_adaptive_mask**: Enable learnable frequency selection
- **use_spectral_norm**: Apply spectral normalization for stability
- **use_multi_scale**: Enable multi-scale temporal processing
- **d_state**: Hidden state dimension
- **expand**: Inner dimension expansion factor

## Architecture

### Core Components

1. **AdaptiveFrequencyS6Block**: Main building block with frequency domain compression
2. **AdaptiveFrequencyMask**: Learnable frequency component selection
3. **AdaptiveFrequencyResidualBlock**: Residual connections with multi-scale processing
4. **CirculantMatrix**: Efficient frequency domain operations

### Frequency Domain Compression

The model compresses state representations by:
1. Converting states to frequency domain via FFT
2. Selecting top-k frequency components
3. Processing in compressed spectral space
4. Reconstructing when needed

## Benchmarks

### LRA Benchmark Support

The model includes comprehensive evaluation tools for the Long Range Arena (LRA) benchmark tasks:

- **ListOps**: Hierarchical mathematical operations
- **Text Classification**: IMDb sentiment analysis  
- **Retrieval**: Document matching tasks
- **Path Finder**: Visual reasoning tasks

Use the evaluation script to test performance on these benchmarks:

```bash
python -m adaptive_frequency_ssm.evaluation \
    --checkpoint_path ./checkpoints/best_model.pt \
    --tasks listops text retrieval pathfinder
```

## Ablation Studies

### Compression Ratio Analysis

The model supports testing different compression ratios to find optimal performance for your use case:

```python
# Test different compression ratios
python -m adaptive_frequency_ssm.evaluation \
    --checkpoint_path ./checkpoints/best_model.pt \
    --compression_ratios 0.1 0.25 0.5 0.75 1.0
```

### Component Analysis

The architecture includes several optional components that can be evaluated:

- **Adaptive Masking**: Learnable frequency component selection
- **Spectral Normalization**: Training stability improvements
- **Multi-scale Processing**: Enhanced performance on varying sequence lengths

## Advanced Usage

### Custom Model Creation

```python
from adaptive_frequency_ssm.models import create_adaptive_frequency_ssm_model

model = create_adaptive_frequency_ssm_model(
    task="classification",
    d_model=768,
    n_layer=24,
    compression_ratio=0.25,
    use_adaptive_mask=True,
    num_classes=10
)
```

### Hybrid Attention Mode

```python
config = AdaptiveFrequencySSMConfig(
    d_model=512,
    n_layer=12,
    use_hybrid_attention=True,
    attention_layers=[3, 7, 11]  # Use attention at these layers
)
```

### Custom Frequency Analysis

```python
from adaptive_frequency_ssm.utils.fft_utils import (
    slice_low_frequencies,
    AdaptiveFrequencyMask,
    multi_scale_fft
)

# Analyze frequency patterns
freq_mask = AdaptiveFrequencyMask(d_state=64, compression_ratio=0.5)
compressed_freqs, selection_mask = freq_mask(hidden_states)
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or sequence length
2. **NaN Loss**: Lower learning rate or enable gradient clipping
3. **Slow Training**: Enable mixed precision (`use_amp: true`)
4. **Poor Convergence**: Adjust warmup steps or compression ratio

### Performance Tips

- Use mixed precision training for potential speedup
- Enable model compilation with PyTorch 2.0+
- Optimize compression ratio for your use case
- Use distributed training for large models

## Citation

```bibtex
@misc{adaptive-frequency-ssm2024,
  title={Adaptive-Frequency-SSM: Advanced Frequency Domain State Space Models},
  author={mini-kio},
  year={2024},
  howpublished={\url{https://github.com/mini-kio/adaptive-frequency-ssm}}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.