# VM-MSP: Vectorized Multilevel Multi-objective Speech Processing

Implementation of the VM-MSP algorithm from "Objective Soups: Multilingual Multi-Task Modeling for Speech Processing" for training multilingual speech recognition and translation models on CoVoST v2 dataset.

## 📊 Overview

VM-MSP (Vectorized Multilevel MSP) is a multi-objective optimization approach that hierarchically organizes training objectives to mitigate gradient conflicts in multilingual multi-task speech processing. This implementation supports:

- **Multiple Languages**: English (en), French (fr), German (de), Spanish (es), Catalan (ca)
- **Multiple Tasks**: Speech Recognition (ASR) and Speech Translation
- **Optimization Orders**: UAS (Unsupervised → ASR → Translation) or USA (Unsupervised → Translation → ASR)
- **Dynamic Weighting**: MoDo algorithm for conflict-avoiding gradient directions
- **Efficient Training**: Automatic detection and selective updating of conflicting layers

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/afmsaif/vm-msp.git
cd vm-msp

# Install dependencies
pip install -r requirements.txt

# Download CoVoST v2 dataset
python download_covost.py
```

### Basic Usage

```python
from vm_msp import VMSPConfig, VMSPTrainer, create_model

# Configure VM-MSP
config = VMSPConfig(
    languages=["en", "fr", "de", "es", "ca"],
    tasks=["asr", "translation"],
    optimization_order="UAS",  # or "USA"
    use_dynamic_weighting=True,
    use_efficient_training=True
)

# Create model and trainer
model = create_model(config)
trainer = VMSPTrainer(model, config)

# Train
trainer.train(train_dataloader, val_dataloader)
```

## 📁 Repository Structure

```
vm-msp/
├── vm_msp.py                 # Main VM-MSP implementation
├── models/
│   ├── conformer.py          # Conformer model architecture
│   ├── whisper_adapter.py    # Whisper model adapter
│   └── wav2vec2_adapter.py   # Wav2Vec2 model adapter
├── data/
│   ├── covost_dataloader.py  # CoVoST v2 data loading
│   └── preprocessing.py      # Audio preprocessing utilities
├── utils/
│   ├── metrics.py            # WER and BLEU evaluation
│   ├── visualization.py      # Training visualization tools
│   └── checkpoint.py         # Model checkpointing
├── configs/
│   ├── uas_config.yaml       # UAS optimization order config
│   └── usa_config.yaml       # USA optimization order config
├── scripts/
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation script
│   └── compare_orders.py     # Compare optimization orders
└── requirements.txt
```

## 🎯 Key Features

### 1. Multilevel Optimization

The algorithm separates objectives into different optimization levels to reduce conflicts:

- **Level 1 (Primary)**: ASR or Translation (depending on order)
- **Level 2 (Secondary)**: Translation or ASR (with penalty η₁)
- **Level 3 (Tertiary)**: Self-supervised learning (with penalty η)

### 2. Dynamic Weight Computation (MoDo)

```python
# The algorithm automatically computes conflict-avoiding weights
# No manual tuning required!
dynamic_weights = ca_optimizer.compute_dynamic_weights(gradients)
```

### 3. Efficient Training Mode

Automatically detects layers with conflicting gradients and focuses computational resources:

```python
config = VMSPConfig(
    use_efficient_training=True,
    conflict_threshold=0.0  # Cosine similarity threshold
)
# Reduces training time by ~17% and memory by ~18%
```

### 4. Optimization Order Selection

Choose between two optimization sequences based on your priority:

#### UAS Order (ASR-focused)
```python
config.optimization_order = "UAS"
# Best for: Applications prioritizing speech recognition accuracy
# Expected: Lower WER, competitive BLEU
```

#### USA Order (Translation-focused)
```python
config.optimization_order = "USA"  
# Best for: Applications prioritizing translation quality
# Expected: Higher BLEU, competitive WER
```

## 📈 Performance Results

Based on the paper's findings with CoVoST v2:

| Method | Avg WER ↓ | Avg BLEU ↑ | Training Time |
|--------|-----------|------------|---------------|
| Two-stage | 25.3% | 22.7 | 3.5h/epoch |
| VS-MSP | 24.6% | 23.1 | 4.1h/epoch |
| VC-MSP | 24.0% | 24.0 | 4.1h/epoch |
| **VM-MSP (UAS)** | **23.6%** | 24.6 | 4.1h/epoch |
| **VM-MSP (USA)** | 23.9% | **24.9** | 4.1h/epoch |
| VM-MSP (Efficient) | 23.6% | 24.8 | 3.7h/epoch |

## 🔧 Configuration Options

### Core Parameters

```python
config = VMSPConfig(
    # Model architecture
    encoder_dim=512,
    num_encoder_layers=12,
    num_attention_heads=8,
    
    # Training hyperparameters
    backbone_lr=5e-5,
    head_lr=5e-4,
    batch_size=32,
    num_epochs=200,
    
    # Penalty parameters
    eta_init=0.0,           # Initial penalty for self-supervised
    eta1_init=0.1,          # Initial penalty for level 2
    eta_increase_rate=0.02,  # Per-epoch increase
    eta_max=1.5,            # Maximum penalty value
    
    # Optimization settings
    optimization_order="UAS",  # "UAS" or "USA"
    use_dynamic_weighting=True,
    modo_lr=0.01,             # Learning rate for MoDo
    
    # Efficiency settings
    use_efficient_training=True,
    conflict_threshold=0.0
)
```

## 🏃 Training Script

### Full Training Pipeline

```bash
# Train with UAS order
python scripts/train.py \
    --config configs/uas_config.yaml \
    --data_path /path/to/covost \
    --output_dir ./outputs/uas \
    --num_gpus 2

# Train with USA order  
python scripts/train.py \
    --config configs/usa_config.yaml \
    --data_path /path/to/covost \
    --output_dir ./outputs/usa \
    --num_gpus 2

# Compare optimization orders
python scripts/compare_orders.py \
    --uas_checkpoint ./outputs/uas/best.pt \
    --usa_checkpoint ./outputs/usa/best.pt \
    --test_data /path/to/test
```

## 📊 Monitoring Training

The implementation includes comprehensive logging and visualization:

```python
# Training metrics are logged to TensorBoard
tensorboard --logdir ./outputs/tensorboard

# Key metrics tracked:
# - Per-language WER and BLEU
# - Per-objective losses
# - Gradient conflict metrics
# - Penalty parameter evolution
# - Layer-wise cosine similarities
```

## 🔬 Advanced Features

### Custom Optimization Order

Define your own optimization hierarchy:

```python
class CustomVMSPTrainer(VMSPTrainer):
    def apply_custom_update(self, gradients, weights, conflicting_layers):
        # Implement your custom optimization order
        # Example: Language-based hierarchy
        pass
```

### Adaptive Penalty Scheduling

```python
# Implement custom penalty scheduling
def adaptive_penalty_schedule(epoch, performance_metrics):
    if performance_metrics['lower_level_degradation'] > threshold:
        return larger_increase_rate
    return standard_increase_rate
```

## 📝 Citation

If you use this implementation, please cite:

```bibtex
@article{saif2024objective,
  title={Objective Soups: Multilingual Multi-Task Modeling for Speech Processing},
  author={Saif, A F M and Chen, Lisha and Cui, Xiaodong and Lu, Songtao and Kingsbury, Brian and Chen, Tianyi},
  journal={Preprint},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
