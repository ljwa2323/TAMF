# Time-aware attention-based deep representation learning for multi-source longitudinal data with structured missingness in electronic medical records

This repository hosts the implementation of our novel framework designed to address structured missingness in multi-source longitudinal electronic medical records (EMR). Traditional methods often fail to fully capture the complexities inherent in EMR data, such as multi-sourced inputs and irregular longitudinal intervals, resulting in significant information loss. Our approach leverages a time-aware attention-based deep representation learning model to overcome these challenges.

### Overview

Our method uniquely handles the temporal attributes of data sources, employing a mask-guided selfattention embedding module to recognize missing patterns and capture longitudinal dependencies within each source. To integrate data from various sources effectively, we designed a time-aware crosssource attention module that aligns embedding sequences chronologically, learning global correlations and time dependencies.

The framework is enhanced with a contrastive loss method that reduces the relative distance between embeddings from different sources, facilitating the fusion of diverse data modalities. Auxiliary tasks leverage missing masks to reconstruct original sequences, aiding in the learning of effective representations between different sources.

![alt text](assets/graphical_abstract.png)

### Key Features
- Mask-Guided Self-Attention Embedding: Captures missing patterns and longitudinal dependencies within each data source.
- Time-Aware Cross-Source Attention Module: Aligns and fuses data from various sources based on chronological order and time dependencies.
- Contrastive Loss Method: Minimizes the distance between different source embeddings, promoting effective data fusion.
- Auxiliary Reconstruction Tasks: Utilizes missing masks for original sequence reconstruction, enhancing fusion representations between sources.

### Installation
```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
```

### Usage

The main training script accepts various command-line arguments to customize the model training:

```bash
python deep_learning/main.py \
    --data_path /path/to/your/data \
    --input_dims 64 32 48 \
    --mask_dims 64 32 48 \
    --time_dims 1 1 1 \
    --embed_dim 256 \
    --num_heads 8 \
    --batch_size 32 \
    --epochs 20 \
    --lr 0.001 \
    --lambda_focal 1.0 \
    --lambda_recon 0.5 \
    --lambda_contrast 0.3 \
    --contrast_margin 1.0 \
    --focal_gamma 2.0 \
    --focal_beta 0.5
```

#### Key Arguments:

- Model Parameters:
  - `--embed_dim`: Dimension of the embedding space (default: 256)
  - `--num_heads`: Number of attention heads (default: 8)
  - `--static_dim`: Dimension of static features (optional)

- Training Parameters:
  - `--batch_size`: Batch size for training (default: 32)
  - `--epochs`: Number of training epochs (default: 20)
  - `--lr`: Learning rate (default: 0.001)

- Loss Function Parameters:
  - `--lambda_focal`: Weight for focal loss (default: 1.0)
  - `--lambda_recon`: Weight for reconstruction loss (default: 0.5)
  - `--lambda_contrast`: Weight for contrastive loss (default: 0.3)
  - `--contrast_margin`: Margin parameter for contrastive loss (default: 1.0)
  - `--focal_gamma`: Gamma parameter for focal loss (default: 2.0)
  - `--focal_beta`: Beta parameter for focal loss (default: 0.5)

- Data Parameters:
  - `--data_path`: Path to the data directory (required)
  - `--input_dims`: Input dimensions for each source (required)
  - `--mask_dims`: Mask dimensions for each source (required)
  - `--time_dims`: Time dimensions for each source (required)

- Other Parameters:
  - `--device`: Device to use for training (default: 'cuda' if available, else 'cpu')
  - `--save_dir`: Directory to save model checkpoints (default: 'checkpoints')
  - `--log_interval`: Print loss every n batches (default: 100)

### Model Checkpoints

The training script automatically saves model checkpoints every 5 epochs in the specified `save_dir`. Each checkpoint contains:
- Model state dict
- Optimizer state dict
- Current epoch
- Current loss

### Logging

Training progress and metrics are logged both to the console and to a log file in the `save_dir`. The log file name includes the timestamp of when training started.

